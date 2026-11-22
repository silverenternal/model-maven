#!/usr/bin/env python3
"""Exercise complex multimodal diagnosis, repair, resume, and export workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def cli(
    binary: Path,
    python: str,
    project: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> dict:
    command = [str(binary), "--python", python, "--json", *args]
    result = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"CLI 未返回单个 JSON 报告：{' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from error
    if result.returncode != 0 or report.get("status") != "success":
        raise RuntimeError(
            f"CLI 操作失败：{' '.join(command)}\n"
            f"{json.dumps(report, ensure_ascii=False, indent=2)}\nstderr:\n{result.stderr}"
        )
    return report


def copy_fixture(source: Path, project: Path) -> None:
    shutil.copytree(source, project / "complex_acceptance")


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="验收复杂多模态模型的调用诊断、局部修复、阶段续训和独立导出。"
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=repository / "target/debug/m-model",
        help="已构建的 m-model 二进制（默认 target/debug/m-model）",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="安装了 PyTorch 的 Python 解释器",
    )
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        parser.error(f"找不到 CLI 二进制：{binary}；先运行 cargo build。")
    torch_check = subprocess.run(
        [args.python, "-c", "import torch"], text=True, capture_output=True
    )
    if torch_check.returncode != 0:
        parser.error(f"{args.python} 环境需要 PyTorch。\n{torch_check.stderr.strip()}")

    source = repository / "examples/complex_acceptance"
    with tempfile.TemporaryDirectory(prefix="model-maven-complex-acceptance-") as temporary:
        root = Path(temporary)
        project = root / "split-training"
        project.mkdir()
        copy_fixture(source, project)
        environment = os.environ.copy()
        pythonpath = [str(root)]
        if environment.get("PYTHONPATH"):
            pythonpath.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath)

        shared_probe = cli(
            binary,
            args.python,
            project,
            "model",
            "probe",
            "--model",
            "complex_acceptance.model:build_model",
            "--batch",
            "complex_acceptance.data:changing_batch",
            env=environment,
        )
        shared_calls = [
            call
            for call in shared_probe["data"]["calls"]
            if call["paths"][0] == "model.fusion.shared_projection"
        ]
        if [call["invocation_index"] for call in shared_calls] != [1, 2]:
            raise RuntimeError("共享融合投影的两次调用没有被分别观测。")

        trial = cli(
            binary,
            args.python,
            project,
            "try",
            "--model",
            "complex_acceptance.model:build_model",
            "--batch",
            "complex_acceptance.data:changing_batch",
            "--step",
            "complex_acceptance.training:multitask_step",
            "--steps",
            "4",
            env=environment,
        )
        if trial["data"]["completed_steps"] != 4:
            raise RuntimeError("复杂多模态 managed-step 试跑未完成四步。")
        losses = trial["data"]["mean_losses"]
        if not all(losses.get(name) is not None for name in ("classification", "alignment", "temporal")):
            raise RuntimeError("试跑没有记录全部三个具名 Loss。")
        metrics = trial["data"]["mean_metrics"]
        if metrics.get("video_frames") != 3.0 or metrics.get("text_tokens") != 4.0:
            raise RuntimeError("试跑没有覆盖两种动态视频/文本长度。")

        repair_model = "complex_acceptance.contract_repair:build_model"
        batch = "complex_acceptance.data:changing_batch"
        source_file = project / "complex_acceptance/contract_repair.py"
        source_digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
        failed = subprocess.run(
            [
                str(binary),
                "--python",
                args.python,
                "--json",
                "model",
                "probe",
                "--model",
                repair_model,
                "--batch",
                batch,
            ],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
        )
        try:
            failure_report = json.loads(failed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"预期中的契约失败没有返回 JSON 报告：\n{failed.stdout}\n{failed.stderr}"
            ) from error
        if failed.returncode == 0 or failure_report.get("status") != "error":
            raise RuntimeError("48→32 维契约错误没有按预期失败。")
        failure = failure_report["data"]["failure_location"]
        failing_call = failure["last_observed_module_call"]
        if (
            failure_report["diagnostics"][0]["code"] != "FORWARD_FAILED"
            or failing_call["paths"][0] != "model.fusion.video_projection"
            or failing_call["inputs"]["items"]["args"]["items"][0]["shape"] != [2, 48]
        ):
            raise RuntimeError("契约失败报告没有定位到 48 维特征与 fusion 投影边界。")

        repaired = cli(
            binary,
            args.python,
            project,
            "layer",
            "set",
            "model.fusion.video_projection",
            "--model",
            repair_model,
            "--batch",
            batch,
            "--param",
            "in_features=48",
            env=environment,
        )
        if repaired["data"]["validation"]["forward"] != "passed":
            raise RuntimeError("修复后的复杂模型没有通过候选前向。")
        if hashlib.sha256(source_file.read_bytes()).hexdigest() != source_digest:
            raise RuntimeError("Layer 修复改写了用户原始 Python 源码。")

        repaired_probe = cli(
            binary,
            args.python,
            project,
            "--variant",
            "working",
            "model",
            "probe",
            "--model",
            repair_model,
            "--batch",
            batch,
            env=environment,
        )
        if repaired_probe["data"]["outputs"]["items"]["logits"]["shape"] != [2, 4]:
            raise RuntimeError("新 CLI 进程没有重载修复变体。")

        uninterrupted = root / "uninterrupted-training"
        copy_fixture(source, uninterrupted)
        shutil.copytree(project / ".model-maven", uninterrupted / ".model-maven")
        common_training = [
            "--variant",
            "working",
            "train",
            "--seed",
            "1234",
            "--training-mode",
            "manual-step",
            "--model",
            repair_model,
            "--batch",
            batch,
            "--step",
            "complex_acceptance.training:manual_multitask_step",
            "--checkpoint-every",
            "1",
        ]
        first = cli(
            binary,
            args.python,
            project,
            *common_training,
            "--steps",
            "2",
            env=environment,
        )
        resumed = cli(
            binary,
            args.python,
            project,
            "train",
            "--resume",
            str(first["data"]["run_id"]),
            "--additional-steps",
            "2",
            "--checkpoint-every",
            "1",
            env=environment,
        )
        full = cli(
            binary,
            args.python,
            uninterrupted,
            *common_training,
            "--steps",
            "4",
            env=environment,
        )
        resumed_data = resumed["data"]
        if (
            resumed_data["completed_steps"] != 4
            or resumed_data["resumed_from_steps"] != 2
            or resumed_data["optimizer_updates_by_group"].get("encoders") != 4
            or resumed_data["optimizer_updates_by_group"].get("fusion_heads") != 4
            or resumed_data["model_state_sha256"] != full["data"]["model_state_sha256"]
            or first["data"]["mean_metrics"].get("video_trainable") != 0.0
            or resumed_data["mean_metrics"].get("video_trainable") != 0.5
            or resumed_data["mean_metrics"].get("video_frames") != 3.0
            or resumed_data["mean_metrics"].get("text_tokens") != 4.0
        ):
            raise RuntimeError("两阶段恢复与同种子四步训练的状态或优化器更新不一致。")
        if resumed_data["resume_state"].get("batch") != "restored":
            raise RuntimeError("恢复没有报告动态批次游标已恢复。")
        if sorted(resumed_data["mean_losses"]) != ["alignment", "classification", "temporal"]:
            raise RuntimeError("正式训练/续训没有保留三个具名 Loss 的累计统计。")

        export_directory = root / "complex-model-export"
        exported = cli(
            binary,
            args.python,
            project,
            "export",
            str(export_directory),
            "--run",
            str(resumed_data["run_id"]),
            env=environment,
        )
        export_manifest = json.loads(
            (export_directory / "manifest.json").read_text(encoding="utf-8")
        )
        if not (export_directory / "variant.json").is_file():
            raise RuntimeError("复杂模型导出包没有包含活动 Layer 变体。")
        expected_hash = resumed_data["model_state_sha256"]
        if export_manifest["checkpoint"].get("model_state_sha256") != expected_hash:
            raise RuntimeError("导出清单中的模型状态指纹与训练 checkpoint 不一致。")
        restored = subprocess.run(
            [args.python, str(export_directory / "restore_model.py")],
            cwd=export_directory,
            text=True,
            capture_output=True,
        )
        if restored.returncode != 0 or expected_hash not in restored.stdout:
            raise RuntimeError(
                "复杂模型导出包没有恢复原模型权重。\n"
                f"stdout:\n{restored.stdout}\nstderr:\n{restored.stderr}"
            )

        print(
            json.dumps(
                {
                    "status": "passed",
                    "shared_module_calls": len(shared_calls),
                    "named_losses": sorted(losses),
                    "dynamic_length_means": {
                        "video_frames": metrics["video_frames"],
                        "text_tokens": metrics["text_tokens"],
                    },
                    "contract_failure_location": failing_call["paths"][0],
                    "local_layer_repair": repaired["data"]["validation"]["forward"],
                    "source_unchanged": True,
                    "reloaded_variant_shape": repaired_probe["data"]["outputs"]["items"]["logits"]["shape"],
                    "resumed_steps": resumed_data["completed_steps"],
                    "resumed_optimizer_groups": resumed_data["optimizer_updates_by_group"],
                    "resumed_matches_uninterrupted": True,
                    "batch_cursor": resumed_data["resume_state"]["batch"],
                    "export_includes_variant": True,
                    "export_restore_state_sha256": export_manifest["checkpoint"]["model_state_sha256"],
                    "limitations": [
                        "离线 CPU 小尺寸夹具证明功能路径，不证明吞吐、数值质量或 GPU/分布式能力。",
                        "不替代真实用户对照研究，也不覆盖所有模型/数据依赖组合。",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"complex acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
