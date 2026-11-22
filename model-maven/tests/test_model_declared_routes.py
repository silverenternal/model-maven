import unittest

import torch

from model_maven_worker.model.loading import _declared_route_evidence


class DeclaredRoutesTest(unittest.TestCase):
    def test_unique_and_shared_registrations_are_reported_without_claiming_flow(self):
        model = torch.nn.Module()
        model.encoders = torch.nn.ModuleDict({"enc_a": torch.nn.Identity(), "enc_b": torch.nn.Identity()})
        model.other = model.encoders["enc_b"]
        model._model_maven_declared_routes = [
            {"slot": "a_encoder", "input": "a", "module_path": "model.encoders.enc_a"},
            {"slot": "b_encoder", "input": "b", "module_path": "model.encoders.enc_b"},
        ]
        report = _declared_route_evidence(model)
        self.assertEqual(report["scope"], "model_self_reported_at_construction")
        self.assertEqual(report["status"], "available")
        self.assertEqual(report["routes"][0]["registered_paths"], ["model.encoders.enc_a"])
        self.assertEqual(set(report["routes"][1]["registered_paths"]), {"model.encoders.enc_b", "model.other"})

    def test_invalid_metadata_is_unavailable_and_does_not_raise(self):
        model = torch.nn.Module()
        self.assertIsNone(_declared_route_evidence(model))
        model._model_maven_declared_routes = [{"slot": "a", "input": "x", "module_path": "model.missing"}]
        self.assertEqual(_declared_route_evidence(model)["status"], "unavailable")
        model._model_maven_declared_routes = [{"slot": "x"}] * 33
        self.assertEqual(_declared_route_evidence(model)["routes"], [])


if __name__ == "__main__":
    unittest.main()
