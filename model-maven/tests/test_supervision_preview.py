import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model_maven_worker.model.structure import module_entries
from model_maven_worker.train.supervision import preview_supervision


class Encoder(nn.Module):
    def forward(self, values):
        return {"features": values * 2}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, values):
        return self.encoder(values)


class SupervisionPreviewTests(unittest.TestCase):
    def test_preview_restores_mixed_per_module_training_modes(self):
        model = Model()
        model.train(True)
        model.encoder.eval()
        self.assertTrue(model.training)
        self.assertFalse(model.encoder.training)
        mark = {
            "id": "latent",
            "binding_status": "current",
            "binding_context": {"model": "fixture:build_model"},
            "module_path": "model.encoder",
            "port": "outputs.features",
            "role": "embedding",
            "rules": [{"id": "bounded", "kind": "range", "mode": "require", "min": -1.0, "max": 1.0, "axis": 0}],
        }
        request = {
            "project_root": tempfile.gettempdir(),
            "model": "fixture:build_model",
        }

        with patch(
            "model_maven_worker.train.supervision.list_marks",
            return_value={"marks": [mark]},
        ):
            report = preview_supervision(
                model,
                request,
                context={"model": "fixture:build_model"},
                torch_module=torch,
                make_batch_fn=lambda *_args: (
                    (torch.tensor([[1.0], [-2.0]]),),
                    {},
                    {
                        "args": [torch.tensor([[1.0], [-2.0]])],
                        "kwargs": {},
                        "metadata": {"stable_sample_ids": True, "sample_ids": ["case-a", "case-b"]},
                    },
                ),
                resolve_entrypoint=lambda _reference: None,
                module_entries_fn=module_entries,
            )

        self.assertEqual(report["results"][0]["status"], "would_reject")
        localized = report["results"][0]["observations"][0]["sample_localization"]
        self.assertEqual(localized["status"], "localized")
        self.assertEqual(
            localized["violating_samples"],
            [{"sample_id": "case-a", "violation_count": 1}, {"sample_id": "case-b", "violation_count": 1}],
        )
        self.assertTrue(report["sample_localization"]["stable_ids_available"])
        self.assertTrue(model.training)
        self.assertFalse(model.encoder.training)

    def test_sample_axis_rule_is_unsupported_without_stable_ids(self):
        model = Model()
        mark = {
            "id": "latent",
            "binding_status": "current",
            "binding_context": {"model": "fixture:build_model"},
            "module_path": "model.encoder",
            "port": "outputs.features",
            "role": "embedding",
            "rules": [{"id": "bounded", "kind": "range", "mode": "require", "min": -1.0, "max": 1.0, "axis": 0}],
        }
        with patch(
            "model_maven_worker.train.supervision.list_marks",
            return_value={"marks": [mark]},
        ):
            report = preview_supervision(
                model,
                {"project_root": tempfile.gettempdir(), "model": "fixture:build_model"},
                context={"model": "fixture:build_model"},
                torch_module=torch,
                make_batch_fn=lambda *_args: (
                    (torch.tensor([[1.0], [-2.0]]),),
                    {},
                    {"args": [torch.tensor([[1.0], [-2.0]])], "kwargs": {}},
                ),
                resolve_entrypoint=lambda _reference: None,
                module_entries_fn=module_entries,
            )
        self.assertEqual(report["results"][0]["status"], "unsupported")
        self.assertIn("没有稳定 sample IDs", report["results"][0]["reason"])
        self.assertEqual(report["sample_localization"]["status"], "insufficient")


if __name__ == "__main__":
    unittest.main()
