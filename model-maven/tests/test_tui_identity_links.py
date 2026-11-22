"""Observed object identity is bounded evidence, not inferred tensor provenance."""
import gc
import unittest

import torch
from model_maven_worker.model.identity_links import MAX_MATCHES, MAX_RELATION_UNITS, TensorIdentityTrace


class TensorIdentityTraceTests(unittest.TestCase):
    def test_matches_same_object_and_version_with_nested_paths(self):
        trace = TensorIdentityTrace(torch)
        tensor = torch.ones(2)
        trace.outputs({"hidden": [tensor]}, "forward:call-2")
        relations = trace.inputs(({"branch": tensor},), {"mask": tensor})
        self.assertEqual([item["input_path"] for item in relations], ["inputs.args[0].branch", "inputs.kwargs.mask"])
        for relation in relations:
            self.assertEqual(relation["prior_outputs"], [{"invocation_id": "forward:call-2", "output_path": "outputs.hidden[0]"}])
        tensor.add_(1)
        self.assertEqual(trace.inputs((tensor,), {})[0]["prior_outputs"], [])
        self.assertEqual(trace.inputs((tensor.clone(),), {})[0]["prior_outputs"], [])

    def test_same_object_multiple_outputs_remains_ambiguous(self):
        trace = TensorIdentityTrace(torch)
        tensor = torch.ones(1)
        trace.outputs(tensor, "call-1")
        trace.outputs(tensor, "call-2")
        self.assertEqual([match["invocation_id"] for match in trace.inputs((tensor,), {})[0]["prior_outputs"]], ["call-1", "call-2"])
        for index in range(MAX_MATCHES + 1):
            trace.outputs(tensor, f"call-extra-{index}")
        self.assertEqual(len(trace.inputs((tensor,), {})[0]["prior_outputs"]), MAX_MATCHES)
        self.assertTrue(trace.coverage()["truncated"])

    def test_weak_references_do_not_retain_inputs_and_ambiguous_keys_are_skipped(self):
        import weakref
        trace = TensorIdentityTrace(torch)
        tensor = torch.ones(1)
        reference = weakref.ref(tensor)
        trace.outputs(tensor, "call-1")
        del tensor
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(trace.inputs((), {"x.y": torch.ones(1)}), [])
        self.assertTrue(trace.coverage()["truncated"])

    def test_relation_budget_marks_coverage_incomplete(self):
        trace = TensorIdentityTrace(torch)
        tensor = torch.ones(1)
        trace.outputs(tensor, "call-1")
        for _ in range(MAX_RELATION_UNITS // 2):
            self.assertEqual(len(trace.inputs((tensor,), {})[0]["prior_outputs"]), 1)
        self.assertEqual(trace.inputs((tensor,), {}), [])
        self.assertTrue(trace.coverage()["truncated"])


if __name__ == "__main__":
    unittest.main()
