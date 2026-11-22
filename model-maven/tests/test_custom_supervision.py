import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from model_maven_worker.protocol import UserError
from model_maven_worker.train.custom_rules import evaluate_plugin, pin_plugin, resolve_plugin


class CustomSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.module = f"custom_supervision_fixture_{id(self)}"
        self.source = self.root / f"{self.module}.py"
        self.source.write_text(
            "def finite(value, batch, params):\n"
            "    return {'passed': bool(value.isfinite().all()), 'summary': {'target': params['target']}}\n"
            "def penalty(value, batch, params):\n"
            "    return (value - params['target']).square().mean()\n",
            encoding="utf-8",
        )

    def tearDown(self):
        sys.modules.pop(self.module, None)
        self.directory.cleanup()

    def test_validator_protocol_pins_source_and_returns_structured_observation(self):
        plugin = pin_plugin(self.root, "custom-validator", f"{self.module}:finite", {"target": "finite"})
        target = resolve_plugin(self.root, plugin, "custom-validator", "require")
        result = evaluate_plugin(
            target, plugin, "custom-validator", torch.tensor([1.0]), {}, torch, training=True
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["custom_summary"], {"target": "finite"})
        self.assertEqual(result["violation_count"], 0)

    def test_penalty_protocol_requires_scalar_differentiable_finite_tensor(self):
        plugin = pin_plugin(self.root, "custom-penalty", f"{self.module}:penalty", {"target": 0.0})
        target = resolve_plugin(self.root, plugin, "custom-penalty", "penalty")
        value = torch.tensor([2.0], requires_grad=True)
        result = evaluate_plugin(target, plugin, "custom-penalty", value, {}, torch, training=True)
        self.assertEqual(result["custom_penalty_value"], 4.0)
        result["penalty_tensor"].backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_changed_plugin_source_fails_closed(self):
        plugin = pin_plugin(self.root, "custom-validator", f"{self.module}:finite", {"target": "finite"})
        self.source.write_text(self.source.read_text() + "# changed\n", encoding="utf-8")
        with self.assertRaises(UserError) as caught:
            resolve_plugin(self.root, plugin, "custom-validator", "observe")
        self.assertEqual(caught.exception.code, "SUPERVISION_PLUGIN_STALE")

    def test_plugin_must_live_inside_project(self):
        outside = tempfile.TemporaryDirectory()
        module = f"outside_supervision_fixture_{id(self)}"
        Path(outside.name, f"{module}.py").write_text("def check(value, batch, params): return {'passed': True}\n")
        sys.path.insert(0, outside.name)
        try:
            with self.assertRaises(UserError) as caught:
                pin_plugin(self.root, "custom-validator", f"{module}:check", {})
            self.assertEqual(caught.exception.code, "SUPERVISION_PLUGIN_SOURCE_UNAVAILABLE")
        finally:
            sys.path.remove(outside.name)
            sys.modules.pop(module, None)
            outside.cleanup()


if __name__ == "__main__":
    unittest.main()
