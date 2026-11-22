import hashlib
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from model_maven_worker.model.semantics import (
    add_from_probe,
    add_rule,
    list_marks,
    list_rules,
    remove_mark,
    remove_rule,
)
from model_maven_worker.protocol import UserError


def _concurrent_add(root, index, gate, results):
    gate.wait()
    try:
        add_rule(root, "latent-feature", kind="range", mode="observe", minimum=index, rule_id=f"concurrent-{index}")
        results.put(None)
    except Exception as error:
        results.put(repr(error))


class SemanticMarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "model.py"
        source.write_text("class Projection: pass\n", encoding="utf-8")
        self.digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.observation = {
            "probe_id": "probe-123",
            "batch_id": "batch-456",
            "batch_identity_stable": False,
            "trace_truncated": False,
            "calls": [{
                "paths": ["model.encoder.projection"],
                "input_output_paths": ["inputs.args[0]", "outputs.features"],
                "invocation_id": "forward-1:call-3",
                "implementation": {
                    "source_file": "model.py",
                    "source_sha256": self.digest,
                    "qualified_type": "project.model.Projection",
                },
            }],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def add(self):
        return add_from_probe(
            self.root,
            self.observation,
            module_path="model.encoder.projection",
            port="outputs.features",
            role="embedding",
            meaning="Project-specific latent representation.",
            mark_id="latent-feature",
            context={
                "model": "fixture:build_model",
                "model_params_sha256": "params",
                "model_injections_sha256": "injections",
                "variant_sha256": None,
                "source_manifest_sha256": "manifest",
            },
        )

    def test_mark_is_bound_to_observed_port_and_survives_reload(self):
        result = self.add()
        self.assertEqual(result["status"], "current")
        report = list_marks(self.root)
        self.assertEqual(report["count"], 1)
        mark = report["marks"][0]
        self.assertEqual(mark["binding_status"], "current")
        self.assertEqual(mark["role"], "embedding")
        self.assertEqual(mark["module_path"], "model.encoder.projection")
        state = json.loads((self.root / result["path"]).read_text(encoding="utf-8"))
        self.assertEqual(state["version"], 1)
        self.assertEqual(state["marks"][0]["anchor"]["source_sha256"], self.digest)

    def test_source_change_marks_binding_stale(self):
        self.add()
        (self.root / "model.py").write_text("class Projection: pass  # changed\n", encoding="utf-8")
        mark = list_marks(self.root)["marks"][0]
        self.assertEqual(mark["binding_status"], "stale")
        self.assertIn("重新 probe", mark["stale_reason"])

    def test_unobserved_port_is_rejected(self):
        self.observation["calls"][0]["input_output_paths"].remove("outputs.features")
        with self.assertRaises(UserError) as error:
            self.add()
        self.assertEqual(error.exception.code, "SEMANTIC_PORT_NOT_OBSERVED")

    def test_truncated_trace_is_rejected(self):
        self.observation["trace_truncated"] = True
        with self.assertRaises(UserError) as error:
            self.add()
        self.assertEqual(error.exception.code, "SEMANTIC_OBSERVATION_TRUNCATED")

    def test_ambiguous_implementation_for_same_path_is_rejected(self):
        second = dict(self.observation["calls"][0])
        second["implementation"] = dict(second["implementation"])
        second["implementation"]["qualified_type"] = "project.other.Projection"
        self.observation["calls"].append(second)
        with self.assertRaises(UserError) as error:
            self.add()
        self.assertEqual(error.exception.code, "SEMANTIC_ANCHOR_AMBIGUOUS")

    def test_concurrent_rule_writers_preserve_all_updates(self):
        self.add()
        context = multiprocessing.get_context("spawn")
        gate = context.Event()
        results = context.Queue()
        processes = [context.Process(target=_concurrent_add, args=(self.root, index, gate, results)) for index in range(12)]
        for process in processes:
            process.start()
        gate.set()
        for process in processes:
            process.join(30)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual([results.get(timeout=5) for _ in processes], [None] * len(processes))
        self.assertEqual({rule["id"] for rule in list_rules(self.root)["rules"]}, {f"concurrent-{index}" for index in range(12)})

    def test_conditional_removal_rejects_rule_replaced_after_confirmation(self):
        self.add()
        add_rule(self.root, "latent-feature", kind="range", mode="observe", minimum=0, rule_id="selected")
        expected = list_rules(self.root)["rules"][0]
        remove_rule(self.root, "selected")
        add_rule(self.root, "latent-feature", kind="range", mode="observe", minimum=10, rule_id="selected")
        with self.assertRaises(UserError) as error:
            remove_rule(self.root, "selected", expected_rule=expected)
        self.assertEqual(error.exception.code, "SUPERVISION_RULE_CHANGED")
        current = list_rules(self.root)["rules"][0]
        self.assertEqual(current["min"], 10)
        remove_rule(self.root, "selected", expected_rule=current)
        self.assertEqual(list_rules(self.root)["count"], 0)

    def test_remove_requires_exact_mark_id(self):
        self.add()
        result = remove_mark(self.root, "latent-feature")
        self.assertEqual(result["count"], 0)
        with self.assertRaises(UserError) as error:
            remove_mark(self.root, "missing")
        self.assertEqual(error.exception.code, "SEMANTIC_MARK_NOT_FOUND")

    def test_malformed_or_escaping_sidecar_is_rejected(self):
        state_path = self.root / ".model-maven/semantic-marks.json"
        state_path.parent.mkdir()
        state_path.write_text(
            json.dumps({"version": 1, "marks": [{"id": "escape", "anchor": {"source_file": "../../outside.py"}}]}),
            encoding="utf-8",
        )
        with self.assertRaises(UserError) as error:
            list_marks(self.root)
        self.assertEqual(error.exception.code, "SEMANTIC_STATE_INVALID")

    def test_explicit_rebind_replaces_stale_anchor_after_a_fresh_observation(self):
        self.add()
        source = self.root / "model.py"
        source.write_text("class Projection: pass  # revised\n", encoding="utf-8")
        self.observation["calls"][0]["implementation"]["source_sha256"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        result = add_from_probe(
            self.root,
            self.observation,
            module_path="model.encoder.projection",
            port="outputs.features",
            role="latent_state",
            meaning="Updated role after checking the new implementation.",
            mark_id="latent-feature",
            replace_existing=True,
            context={
                "model": "fixture:build_model",
                "model_params_sha256": "params",
                "model_injections_sha256": "injections",
                "variant_sha256": None,
                "source_manifest_sha256": "manifest-v2",
            },
        )
        self.assertEqual(result["status"], "rebound")
        mark = list_marks(self.root)["marks"][0]
        self.assertEqual(mark["binding_status"], "current")
        self.assertEqual(mark["role"], "latent_state")
        self.assertEqual(mark["anchor"]["source_sha256"], self.observation["calls"][0]["implementation"]["source_sha256"])

    def test_rule_crud_keeps_explicit_mode_and_bound_mark(self):
        self.add()
        saved = add_rule(
            self.root,
            "latent-feature",
            kind="range",
            mode="require",
            minimum=-1.0,
            maximum=1.0,
            rule_id="bounded-latent",
        )
        self.assertEqual(saved["rule"]["mode"], "require")
        listed = list_rules(self.root)
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["rules"][0]["mark_id"], "latent-feature")
        self.assertEqual(remove_rule(self.root, "bounded-latent")["mark_id"], "latent-feature")
        self.assertEqual(list_rules(self.root)["count"], 0)

    def test_rule_rejects_implicit_or_invalid_penalties(self):
        self.add()
        with self.assertRaises(UserError) as error:
            add_rule(self.root, "latent-feature", kind="range", mode="penalty", minimum=0.0)
        self.assertEqual(error.exception.code, "SUPERVISION_RULE_VALUE_INVALID")
        with self.assertRaises(UserError) as error:
            add_rule(self.root, "latent-feature", kind="finite", mode="penalty", weight=0.2)
        self.assertEqual(error.exception.code, "SUPERVISION_RULE_VALUE_INVALID")

    def test_stale_mark_cannot_receive_a_new_rule(self):
        self.add()
        (self.root / "model.py").write_text("class Projection: pass  # changed\n", encoding="utf-8")
        with self.assertRaises(UserError) as error:
            add_rule(self.root, "latent-feature", kind="finite", mode="observe")
        self.assertEqual(error.exception.code, "SEMANTIC_MARK_STALE")


if __name__ == "__main__":
    unittest.main()
