#!/usr/bin/env python3
"""Exercise the human-authored semantic mark and explicit supervision workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


FIXTURE = r'''import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, values):
        return {"features": values * 2 * self.scale}

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, values, labels=None):
        return self.encoder(values)

def build_model():
    return Model()

_BATCH_CALL = 0

def matching_batch():
    global _BATCH_CALL
    first = _BATCH_CALL * 2
    _BATCH_CALL += 1
    return {"args": [torch.tensor([[0.25], [0.5]])], "kwargs": {"labels": torch.tensor([0, 1])}, "metadata": {"stable_sample_ids": True, "sample_ids": [f"fixture:sample-{first}", f"fixture:sample-{first + 1}"]}}

def mismatched_batch():
    return {"args": [torch.tensor([[0.25], [0.5]])], "kwargs": {"labels": torch.tensor([0, 1, 2])}, "metadata": {"stable_sample_ids": True, "sample_ids": ["fixture:sample-0", "fixture:sample-1"]}}

def plain_batch():
    return {"args": [torch.tensor([[0.25], [0.5]])], "kwargs": {"labels": torch.tensor([0, 1])}}

def supervised_loop(model, context):
    optimizer = getattr(supervised_loop, "optimizer", None)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        pending = getattr(supervised_loop, "pending_optimizer_state", None)
        if pending is not None:
            optimizer.load_state_dict(pending["optimizer"])
        supervised_loop.optimizer = optimizer
        supervised_loop.position = pending.get("position", 0) if pending else 0
    context.register_optimizer(optimizer)
    for step in range(context.remaining_steps):
        position = supervised_loop.position
        values = torch.tensor([[0.25], [0.5]])
        batch = {"args": [values], "kwargs": {"labels": torch.tensor([0, 1])}, "metadata": {
            "stable_sample_ids": True,
            "sample_ids": [f"external:sample-{position * 2}", f"external:sample-{position * 2 + 1}"],
        }}
        optimizer.zero_grad(set_to_none=True)
        with context.supervised_step(batch):
            output = model(values, labels=batch["kwargs"]["labels"])
            loss = output["features"].sum() * 0
            total_loss = context.backward(loss)
            optimizer.step()
        supervised_loop.position = position + 1
        if not context.report_step(total_loss.detach()):
            break

def _save_supervised_loop_state():
    return {"optimizer": supervised_loop.optimizer.state_dict(), "position": supervised_loop.position}

def _load_supervised_loop_state(state):
    supervised_loop.pending_optimizer_state = state

supervised_loop.state_dict = _save_supervised_loop_state
supervised_loop.load_state_dict = _load_supervised_loop_state
'''

TRAINING = r'''def step(model, batch, context):
    output = model(batch["args"][0], labels=batch["kwargs"]["labels"])
    return {"loss": output["features"].sum() * 0}
'''

REPORTED_IDS_TRAINING = r'''def step(model, batch, context):
    context.report_sample_ids(["fixture:sample-0", "fixture:sample-1"])
    output = model(batch["args"][0], labels=batch["kwargs"]["labels"])
    return {"loss": output["features"].sum() * 0}
'''

CUSTOM_RULES = r'''import torch

def bounded(value, batch, params):
    violations = value.abs() > params["limit"]
    return {
        "passed": not bool(violations.any().item()),
        "summary": {"limit": params["limit"], "max_abs": float(value.detach().abs().max().item())},
        "violations": violations,
    }

def energy(value, batch, params):
    return (value - params.get("target", 0.0)).square().mean()
'''


def call(binary: Path, python: str, root: Path, args: list[str], *, expect_success: bool):
    command = [str(binary), "--json", "--python", python, *args]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"command emitted invalid JSON: {command!r}\nstdout={result.stdout}\nstderr={result.stderr}"
        ) from error
    if expect_success and result.returncode != 0:
        raise RuntimeError(f"expected success: {command!r}\n{report!r}\n{result.stderr}")
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f"expected failure: {command!r}\n{report!r}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="model-maven-supervision-acceptance-") as temporary:
        root = Path(temporary)
        (root / "fixture.py").write_text(FIXTURE, encoding="utf-8")
        (root / "training.py").write_text(TRAINING, encoding="utf-8")
        (root / "reported_training.py").write_text(REPORTED_IDS_TRAINING, encoding="utf-8")
        (root / "constraints.py").write_text(CUSTOM_RULES, encoding="utf-8")

        mark = call(
            binary,
            args.python,
            root,
            [
                "model", "mark", "--model", "fixture:build_model", "--batch", "fixture:matching_batch",
                "--path", "model.encoder", "--port", "outputs.features", "--role", "representation",
                "--meaning", "Features consumed by the next model block.", "--id", "representation",
            ],
            expect_success=True,
        )
        assert mark["data"]["mark"]["role"] == "representation", mark

        rule = call(
            binary,
            args.python,
            root,
            [
                "model", "supervise", "rule", "add", "--mark", "representation", "--kind",
                "batch-alignment", "--mode", "require", "--target-path", "batch.kwargs.labels",
                "--axis", "0", "--id", "batch-size",
            ],
            expect_success=True,
        )
        assert rule["data"]["rule"]["target_path"] == "batch.kwargs.labels", rule

        range_rule = call(
            binary,
            args.python,
            root,
            [
                "model", "supervise", "rule", "add", "--mark", "representation", "--kind",
                "range", "--mode", "observe", "--max", "0.75", "--axis", "0", "--id", "sample-range",
            ],
            expect_success=True,
        )
        assert range_rule["data"]["rule"]["axis"] == 0, range_rule
        preview = call(
            binary,
            args.python,
            root,
            [
                "model", "supervise", "preview", "--model", "fixture:build_model",
                "--batch", "fixture:matching_batch",
            ],
            expect_success=True,
        )
        assert preview["data"]["objective_applied"] is False, preview
        assert preview["data"]["coverage_status"] == "partial", preview
        assert preview["data"]["sample_localization"]["status"] == "reported", preview
        assert preview["data"]["results"][0]["status"] == "passed", preview
        localized = next(result for result in preview["data"]["results"] if result["rule_id"] == "sample-range")
        localized_samples = localized["observations"][0]["sample_localization"]["violating_samples"]
        assert localized_samples == [{"sample_id": "fixture:sample-1", "violation_count": 1}], preview

        trained = call(
            binary,
            args.python,
            root,
            [
                "train", "--model", "fixture:build_model", "--batch", "fixture:matching_batch",
                "--step", "training:step", "--steps", "2", "--checkpoint-every", "1",
                "--apply-supervision",
            ],
            expect_success=True,
        )
        supervision = trained["data"]["supervision"]
        assert supervision["status"] == "applied", trained
        assert supervision["coverage_status"] == "partial", trained
        assert supervision["sample_localization"]["status"] == "reported", trained
        evidence = supervision["rules"][0]["last_target_evidence"]
        assert evidence["matched"] is True and evidence["step"] == 2, trained
        range_evidence = next(rule for rule in supervision["rules"] if rule["rule_id"] == "sample-range")
        training_violations = [{"sample_id": "fixture:sample-3", "violation_count": 1}]
        assert range_evidence["last_sample_localization"]["violating_samples"] == training_violations, trained
        initial_coverage = range_evidence["sample_coverage"]
        assert initial_coverage["localized_batches"] == 2, trained
        assert initial_coverage["sample_occurrences"] == 4, trained
        assert initial_coverage["unique_sample_count_lower_bound"] == 4, trained

        resumed = call(
            binary,
            args.python,
            root,
            [
                "train", "--resume", trained["data"]["run_id"], "--additional-steps", "2",
                "--checkpoint-every", "1", "--allow-state-reset", "--apply-supervision",
            ],
            expect_success=True,
        )
        resumed_coverage = next(
            rule for rule in resumed["data"]["supervision"]["rules"] if rule["rule_id"] == "sample-range"
        )["sample_coverage"]
        assert resumed["data"]["resumed_from_steps"] == 2, resumed
        assert resumed["data"]["completed_steps"] == 4, resumed
        assert resumed_coverage["localized_batches"] == 4, resumed
        assert resumed_coverage["sample_occurrences"] == 8, resumed
        assert resumed_coverage["unique_sample_count_lower_bound"] == 4, resumed
        assert resumed_coverage["unique_sample_count_complete"] is True, resumed
        assert resumed_coverage["unique_violating_sample_count_lower_bound"] == 2, resumed

        external_penalty_rule = call(
            binary,
            args.python,
            root,
            [
                "model", "supervise", "rule", "add", "--mark", "representation", "--kind", "range",
                "--mode", "penalty", "--max", "0.75", "--weight", "1", "--axis", "0",
                "--id", "external-penalty",
            ],
            expect_success=True,
        )
        assert external_penalty_rule["data"]["rule"]["mode"] == "penalty", external_penalty_rule
        external = call(
            binary,
            args.python,
            root,
            [
                "train", "--training-mode", "external-loop", "--model", "fixture:build_model",
                "--loop", "fixture:supervised_loop", "--steps", "2", "--checkpoint-every", "1",
                "--apply-supervision",
            ],
            expect_success=True,
        )
        external_supervision = external["data"]["supervision"]
        assert external_supervision["training_mode"] == "external-loop", external
        external_rule = next(
            rule for rule in external_supervision["rules"] if rule["rule_id"] == "sample-range"
        )
        assert external_rule["sample_coverage"]["localized_batches"] == 2, external
        assert external_rule["sample_coverage"]["unique_sample_count_lower_bound"] == 4, external
        assert external["data"]["mean_metrics"]["supervision/external-penalty"] > 0, external
        external_resumed = call(
            binary,
            args.python,
            root,
            [
                "train", "--resume", external["data"]["run_id"],
                "--additional-steps", "2", "--checkpoint-every", "1", "--apply-supervision",
            ],
            expect_success=True,
        )
        external_coverage = next(
            rule for rule in external_resumed["data"]["supervision"]["rules"]
            if rule["rule_id"] == "sample-range"
        )["sample_coverage"]
        assert external_resumed["data"]["completed_steps"] == 4, external_resumed
        assert external_coverage["localized_batches"] == 4, external_resumed
        assert external_coverage["sample_occurrences"] == 8, external_resumed
        assert external_coverage["unique_sample_count_lower_bound"] == 8, external_resumed
        assert resumed_coverage["unique_violating_sample_count_complete"] is True, resumed
        assert resumed_coverage["unlocalized_batches"] == 0, resumed
        checkpoint_dir = root / ".model-maven" / "runs" / trained["data"]["run_id"] / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert checkpoints, f"expected persisted checkpoints under {checkpoint_dir}"
        assert all(
            b"fixture:sample-" not in checkpoint.read_bytes()
            for checkpoint in checkpoints
        ), "raw sample IDs must not be stored in checkpoint payloads"

        require_rule = call(
            binary,
            args.python,
            root,
            [
                "model", "supervise", "rule", "add", "--mark", "representation", "--kind",
                "range", "--mode", "require", "--max", "0.75", "--axis", "0", "--id", "sample-require",
            ],
            expect_success=True,
        )
        assert require_rule["data"]["rule"]["axis"] == 0, require_rule

        rejected_sample = call(
            binary,
            args.python,
            root,
            [
                "try", "--model", "fixture:build_model", "--batch", "fixture:matching_batch",
                "--step", "training:step", "--steps", "1", "--apply-supervision",
            ],
            expect_success=False,
        )
        cause = rejected_sample["data"]["cause_data"]
        assert rejected_sample["data"]["cause_code"] == "SUPERVISION_REQUIRE_FAILED", rejected_sample
        assert cause["sample_localization"]["status"] == "localized", rejected_sample
        assert cause["sample_localization"]["violating_samples"] == localized_samples, rejected_sample
        assert rejected_sample["data"]["completed_steps"] == 0, rejected_sample

        external_rejected = call(
            binary,
            args.python,
            root,
            [
                "try", "--training-mode", "external-loop", "--model", "fixture:build_model",
                "--loop", "fixture:supervised_loop", "--steps", "1", "--apply-supervision",
            ],
            expect_success=False,
        )
        assert external_rejected["diagnostics"][0]["code"] == "SUPERVISION_REQUIRE_FAILED", external_rejected
        assert external_rejected["data"]["sample_localization"]["violating_samples"] == [
            {"sample_id": "external:sample-1", "violation_count": 1}
        ], external_rejected

        rejected_reported_ids = call(
            binary,
            args.python,
            root,
            [
                "try", "--model", "fixture:build_model", "--batch", "fixture:plain_batch",
                "--step", "reported_training:step", "--steps", "1", "--apply-supervision",
            ],
            expect_success=False,
        )
        assert rejected_reported_ids["data"]["cause_data"]["sample_localization"]["status"] == "localized", rejected_reported_ids
        assert rejected_reported_ids["data"]["cause_data"]["sample_localization"]["violating_samples"] == localized_samples, rejected_reported_ids
        assert rejected_reported_ids["data"]["completed_steps"] == 0, rejected_reported_ids

        rejected = call(
            binary,
            args.python,
            root,
            [
                "try", "--model", "fixture:build_model", "--batch", "fixture:mismatched_batch",
                "--step", "training:step", "--steps", "1", "--apply-supervision",
            ],
            expect_success=False,
        )
        assert rejected["data"]["cause_code"] == "SUPERVISION_REQUIRE_FAILED", rejected
        assert rejected["data"]["cause_data"]["target_shape"] == [3], rejected
        assert rejected["data"]["completed_steps"] == 0, rejected

        custom_validator = call(
            binary,
            args.python,
            root,
            [
                "model", "supervise", "rule", "add", "--mark", "representation", "--kind",
                "custom-validator", "--mode", "observe", "--plugin", "constraints:bounded",
                "--plugin-param", "limit=0.75", "--axis", "0", "--id", "custom-bounds",
            ],
            expect_success=True,
        )
        assert custom_validator["data"]["rule"]["kind"] == "custom-validator", custom_validator
        custom_penalty = call(
            binary,
            args.python,
            root,
            [
                "model", "supervise", "rule", "add", "--mark", "representation", "--kind",
                "custom-penalty", "--mode", "penalty", "--plugin", "constraints:energy",
                "--plugin-param", "target=0.0", "--weight", "0.25", "--id", "custom-energy",
            ],
            expect_success=True,
        )
        assert custom_penalty["data"]["rule"]["mode"] == "penalty", custom_penalty
        custom_preview = call(
            binary,
            args.python,
            root,
            [
                "model", "supervise", "preview", "--model", "fixture:build_model",
                "--batch", "fixture:matching_batch",
            ],
            expect_success=True,
        )
        validator_result = next(
            item for item in custom_preview["data"]["results"] if item["rule_id"] == "custom-bounds"
        )
        assert validator_result["status"] == "observed_violation", custom_preview
        assert validator_result["observations"][0]["custom_summary"]["limit"] == 0.75, custom_preview
        penalty_preview = next(
            item for item in custom_preview["data"]["results"] if item["rule_id"] == "custom-energy"
        )
        assert penalty_preview["objective_applied"] is False, custom_preview
        assert penalty_preview["observations"][0]["weighted_penalty_preview"] > 0, custom_preview
        call(
            binary,
            args.python,
            root,
            ["model", "supervise", "rule", "remove", "--id", "sample-require"],
            expect_success=True,
        )
        custom_trained = call(
            binary,
            args.python,
            root,
            [
                "try", "--model", "fixture:build_model", "--batch", "fixture:matching_batch",
                "--step", "training:step", "--steps", "1", "--apply-supervision",
            ],
            expect_success=True,
        )
        assert custom_trained["data"]["mean_losses"]["supervision/custom-energy"] > 0, custom_trained

        print(json.dumps({
            "status": "verified",
            "mark": mark["data"]["mark"]["id"],
            "rule": rule["data"]["rule"]["id"],
            "preview": preview["data"]["results"][0]["status"],
            "training_step": evidence,
            "resumed_sample_coverage": resumed_coverage,
        "external_loop_resume_coverage": external_coverage,
        "external_loop_supervision": external_supervision["training_mode"],
        "checkpoint_contains_raw_sample_ids": False,
            "mismatch": {
                "cause_code": rejected["data"]["cause_code"],
                "completed_steps": rejected["data"]["completed_steps"],
            },
            "sample_rule_rejection": {
                "cause_code": rejected_sample["data"]["cause_code"],
                "localized_violations": cause["sample_localization"]["violating_samples"],
                "completed_steps": rejected_sample["data"]["completed_steps"],
            },
            "context_reported_ids": {
                "localized_violations": rejected_reported_ids["data"]["cause_data"]["sample_localization"]["violating_samples"],
                "completed_steps": rejected_reported_ids["data"]["completed_steps"],
            },
            "coverage_status": supervision["coverage_status"],
            "sample_localization": supervision["sample_localization"],
            "custom_validator": validator_result["observations"][0]["custom_summary"],
            "custom_penalty_loss": custom_trained["data"]["mean_losses"]["supervision/custom-energy"],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
