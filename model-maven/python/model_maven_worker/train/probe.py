"""Training operation preflight, source evidence and runtime provenance."""

from dataclasses import dataclass
from typing import Any, Callable

from ..protocol import UserError
from .state import checkpoint_environment_inventory


@dataclass(frozen=True)
class TrainingProbeRuntime:
    environment_snapshot: Callable[[str], dict]
    entrypoint: Callable[[str], Any]
    source_manifest: Callable[[str], dict]
    run_training_run: Callable[[Any, dict, dict], dict]
    run_training_probe: Callable[[Any, dict], dict]
    python_executable: str
    python_version: str
    pytorch_version: str


@dataclass(frozen=True)
class TrainingModeRuntime:
    run_managed_steps: Callable[..., dict]
    run_manual_steps: Callable[..., dict]
    run_external_loop: Callable[..., dict]


def run_training_mode(runtime, model, request):
    mode = request.get("training_mode", "managed-step")
    if mode == "managed-step":
        return runtime.run_managed_steps(model, request)
    if mode == "manual-step":
        return runtime.run_manual_steps(model, request)
    if mode == "external-loop":
        return runtime.run_external_loop(model, request)
    raise UserError("TRAINING_MODE_UNSUPPORTED", f"当前不支持训练模式：{mode}")


def training_probe_with_source_evidence(runtime, model, request):
    external_loop = request.get("training_mode") == "external-loop"
    if external_loop:
        if request.get("batch"):
            raise UserError("EXTERNAL_LOOP_BATCH_UNUSED", "external-loop 自己管理数据与批次，不接受独立 --batch 入口。")
        if not request.get("training_loop"):
            raise UserError("EXTERNAL_LOOP_REQUIRED", "external-loop 缺少用户循环入口。", "使用 --loop module:run_loop。")
    else:
        if not request.get("batch"):
            raise UserError("BATCH_REQUIRED", "训练试跑需要一个批次工厂。", "使用 --batch module:batch_factory。")
        if not request.get("training_step"):
            raise UserError("STEP_REQUIRED", "训练试跑需要一个用户步骤。", "使用 --step module:training_step。")
    training_environment = None
    environment_fingerprint_sha256 = None
    checkpoint_inventory = None
    capture_timing = "training_start" if request.get("operation") == "train" else "probe_start"
    try:
        current_environment = runtime.environment_snapshot(capture_timing)
        environment_fingerprint_sha256 = current_environment["fingerprint_sha256"]
        checkpoint_inventory = checkpoint_environment_inventory(current_environment)
        if request.get("operation") == "train":
            training_environment = current_environment
    except Exception as error:
        # Environment capture is provenance, not a prerequisite for new work.
        if request.get("operation") == "train":
            training_environment = {
                "status": "unavailable",
                "capture_timing": capture_timing,
                "diagnostics": [f"{type(error).__name__}: {error}"],
            }
        expected_runtime = request.get("expected_runtime")
        if isinstance(expected_runtime, dict) and expected_runtime.get("environment_fingerprint_sha256"):
            raise UserError(
                "RESUME_ENVIRONMENT_UNVERIFIABLE",
                "无法核对原运行的 Python distribution 清单，本次重放/续训未执行。",
                "恢复可读取完整包元数据的原 Python 环境后重试。",
                {"cause": f"{type(error).__name__}: {error}"},
            ) from error
    request["_environment_fingerprint_sha256"] = environment_fingerprint_sha256
    request["_checkpoint_environment_inventory"] = checkpoint_inventory
    # Import explicit entrypoints before hashing, so the starting evidence covers
    # the user code that the bounded run is about to call.
    if external_loop:
        runtime.entrypoint(request.get("training_loop"))
    else:
        runtime.entrypoint(request.get("batch"))
        runtime.entrypoint(request.get("training_step"))
    before = runtime.source_manifest(request["project_root"])

    def attach_evidence(target):
        try:
            after = runtime.source_manifest(request["project_root"])
        except UserError:
            after = {}
        changed = sorted(
            path
            for path in before.keys() & after.keys()
            if before[path] != after[path]
        )
        removed = sorted(before.keys() - after.keys())
        added = sorted(after.keys() - before.keys())
        target["source_manifest"] = before
        target["source_manifest_after"] = after
        target["source_changed_during_run"] = bool(changed or removed or added)
        target["source_hash_changes"] = changed
        target["source_files_missing_after"] = removed
        target["source_files_newly_observed"] = added
        if request.get("code_snapshot") is not None:
            target["code_snapshot"] = request["code_snapshot"]
        if training_environment is not None:
            target["environment_snapshot"] = training_environment
        target["runtime"] = {
            "python_executable": runtime.python_executable,
            "python_version": runtime.python_version,
            "pytorch_version": str(runtime.pytorch_version),
            **request.get("_device_report", {"device": "cpu", "device_backend": "cpu", "device_name": "CPU"}),
        }
        if request.get("_resource_report") is not None:
            target["resources"] = request["_resource_report"]
        if environment_fingerprint_sha256:
            target["runtime"]["environment_fingerprint_sha256"] = environment_fingerprint_sha256
        return target

    try:
        operation = request.get("operation")
        source_error_code = "RESUME_SOURCE_CHANGED" if operation == "train" else "REPEAT_SOURCE_CHANGED"
        source_error_message = (
            "checkpoint 对应的项目源码已变化，本次恢复在训练步骤前停止。"
            if operation == "train"
            else "原运行记录中的项目源码已变化，本次重跑在执行训练步骤前停止。"
        )
        expected_manifest = request.get("expected_source_manifest")
        if expected_manifest is not None and before != expected_manifest:
            expected = expected_manifest if isinstance(expected_manifest, dict) else {}
            changed = sorted(
                path for path in before.keys() & expected.keys() if before[path] != expected[path]
            )
            raise UserError(
                source_error_code,
                source_error_message,
                "查看 changed_files；恢复训练请回到原源码，重跑试验请用当前代码启动新的 try。",
                {
                    "changed_files": changed,
                    "missing_files": sorted(expected.keys() - before.keys()),
                    "new_files": sorted(before.keys() - expected.keys()),
                },
            )
        expected_runtime = request.get("expected_runtime")
        if isinstance(expected_runtime, dict):
            runtime_changes = {}
            if expected_runtime.get("python_version") not in (None, runtime.python_version):
                runtime_changes["python_version"] = {
                    "expected": expected_runtime["python_version"],
                    "actual": runtime.python_version,
                }
            if expected_runtime.get("pytorch_version") not in (None, runtime.pytorch_version):
                runtime_changes["pytorch_version"] = {
                    "expected": expected_runtime["pytorch_version"],
                    "actual": runtime.pytorch_version,
                }
            expected_fingerprint = expected_runtime.get("environment_fingerprint_sha256")
            if expected_fingerprint and expected_fingerprint != environment_fingerprint_sha256:
                runtime_changes["environment_fingerprint_sha256"] = {
                    "expected": expected_fingerprint,
                    "actual": environment_fingerprint_sha256,
                }
            if runtime_changes:
                raise UserError(
                    "RESUME_RUNTIME_CHANGED" if operation == "train" else "REPEAT_RUNTIME_CHANGED",
                    "原运行使用的 Python/PyTorch 或安装包清单与当前环境不一致，本次操作未执行训练步骤。",
                    "恢复原运行的 Python 环境后重试，或在新环境下建立新的运行记录。",
                    {"runtime_changes": runtime_changes},
                )
        if request.get("operation") == "train":
            result = runtime.run_training_run(model, request, before)
        else:
            result = runtime.run_training_probe(model, request)
    except UserError as error:
        attach_evidence(error.data)
        if error.data.get("source_changed_during_run"):
            raise UserError(
                "SOURCE_CHANGED_DURING_RUN",
                "运行期间项目源码发生变化；本次结果已标记失败，不能当作基于单一代码版本的有效结果。",
                "恢复运行开始时的源码后再续训，或用当前源码启动新的运行。详情包含变化文件和运行前后源码指纹。",
                error.data,
            ) from error
        raise
    attach_evidence(result)
    if result.get("source_changed_during_run"):
        evidence = {
            key: result.get(key)
            for key in (
                "source_manifest",
                "source_manifest_after",
                "source_changed_during_run",
                "source_hash_changes",
                "source_files_missing_after",
                "source_files_newly_observed",
                "code_snapshot",
                "completed_steps",
                "mean_loss",
                "optimizer_updates",
            )
            if key in result
        }
        raise UserError(
            "SOURCE_CHANGED_DURING_RUN",
            "运行期间项目源码发生变化；本次结果已标记失败，不能当作基于单一代码版本的有效结果。",
            "恢复运行开始时的源码后再续训，或用当前源码启动新的运行。详情包含变化文件和运行前后源码指纹。",
            evidence,
        )
    return result
