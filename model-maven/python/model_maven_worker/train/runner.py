"""Formal training run orchestration and checkpoint lifecycle."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError
from ..run.provenance import check_snapshot_project_imports
from .state import callable_checkpoint_context


@dataclass(frozen=True)
class TrainingRunRuntime:
    entrypoint: Callable[[str], Any]
    load_training_checkpoint: Callable[..., dict]
    load_verified_checkpoint_payload: Callable[..., tuple[dict, dict]]
    migrate_state_dict: Callable[..., tuple[dict, dict]]
    state_dict_digest: Callable[..., str]
    model_state_alias_groups: Callable[[Any], Any]
    restore_callable_checkpoint: Callable[..., str]
    restore_rng_state: Callable[[dict], None]
    training_run_signature: Callable[[dict], dict]
    checkpoint_value_supported: Callable[[Any], bool]
    callable_checkpoint_hooks: Callable[..., Any]
    checkpoint_rng_state: Callable[[], dict]
    model_state_digest: Callable[[Any], str]
    save_training_checkpoint: Callable[..., dict]
    run_external_loop: Callable[..., dict]
    run_managed_steps: Callable[..., dict]
    run_manual_steps: Callable[..., dict]
    emit_progress: Callable[[dict], None]
    python_version: str
    pytorch_version: str


def run_training_run(runtime, model, request, source_manifest):
    mode = request.get("training_mode", "managed-step")
    if mode == "external-loop":
        if request.get("batch"):
            raise UserError("EXTERNAL_LOOP_BATCH_UNUSED", "external-loop 自己管理数据与批次，不接受独立 --batch 入口。")
        if not request.get("training_loop"):
            raise UserError("EXTERNAL_LOOP_REQUIRED", "external-loop 缺少用户循环入口。", "使用 --loop module:run_loop。")
    elif not request.get("batch") or not request.get("training_step"):
        raise UserError("TRAIN_INPUTS_REQUIRED", "正式 managed/manual 训练需要模型、批次工厂和 training_step。")
    steps = request.get("steps")
    checkpoint_every = request.get("checkpoint_every")
    if type(steps) is not int or not 1 <= steps <= 1_000_000:
        raise UserError("TRAINING_PARAMETER_INVALID", "正式训练总步骤数必须在 1..1,000,000。")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise UserError("TRAINING_PARAMETER_INVALID", "checkpoint 间隔必须是正整数。")
    directory = request.get("checkpoint_dir")
    if not directory:
        raise UserError("CHECKPOINT_PATH_INVALID", "正式训练缺少运行专属 checkpoint 目录。")
    if mode not in {"managed-step", "manual-step", "external-loop"}:
        raise UserError("TRAINING_MODE_UNSUPPORTED", f"当前不支持正式训练模式：{mode}")
    migration_source = request.get("migration_checkpoint_input")
    migration_plan = request.get("migration_plan")
    if (migration_source is None) != (migration_plan is None):
        raise UserError(
            "MIGRATION_REQUEST_INVALID",
            "权重迁移必须同时提供父 checkpoint 和显式迁移计划。",
        )
    if migration_source is not None and (request.get("checkpoint_input") or model is None):
        raise UserError(
            "MIGRATION_REQUEST_INVALID",
            "结构迁移只能为新子运行执行，且必须提供目标模型；不能与普通 resume 混用。",
        )
    loop_fn = runtime.entrypoint(request["training_loop"]) if mode == "external-loop" else None
    step_fn = runtime.entrypoint(request["training_step"]) if mode != "external-loop" else None
    batch_fn = runtime.entrypoint(request["batch"]) if mode != "external-loop" else None
    callable_contexts = {}
    selected_callables = (
        {"loop": (loop_fn, "external-loop")}
        if mode == "external-loop"
        else {"batch": (batch_fn, "batch 工厂"), "step": (step_fn, "training_step")}
    )
    for slot, (function, label) in selected_callables.items():
        context = callable_checkpoint_context(function, label)
        if context is not None:
            callable_contexts[slot] = context
    request["_checkpoint_callable_contexts"] = callable_contexts
    restore = None
    batch_state_status = "new"
    step_state_status = "new"
    loop_state_status = "new"
    checkpoint_input = request.get("checkpoint_input")
    if checkpoint_input:
        restore = runtime.load_training_checkpoint(checkpoint_input, model, request, source_manifest)
        allow_reset = bool(request.get("allow_state_reset"))
        if mode == "external-loop":
            loop_state_status = runtime.restore_callable_checkpoint(
                loop_fn, restore.get("loop_state"), "external-loop", allow_reset
            )
            if (
                isinstance(restore.get("loop_state"), dict)
                and restore["loop_state"].get("kind") == "versioned-hooks"
                and loop_state_status in {"restored", "migrated"}
            ):
                # ExternalLoopContext exposes the restored author state as a
                # transient plain hook payload; the next durable checkpoint
                # writes the validated versioned envelope again.
                restore["loop_state"] = {
                    "kind": "hooks",
                    "state": loop_fn.state_dict(),
                }
        else:
            batch_state_status = runtime.restore_callable_checkpoint(
                batch_fn, restore.get("batch_state"), "batch 工厂", allow_reset
            )
            step_state_status = runtime.restore_callable_checkpoint(
                step_fn, restore.get("step_state"), "training_step", allow_reset
            )
        runtime.restore_rng_state(restore.get("rng_state"))
    else:
        restore = {"completed_steps": 0, "engine_state": {}, "loop_state": None}
    start_step = int(restore.get("completed_steps", 0))
    if steps <= start_step:
        raise UserError("TRAINING_TARGET_REACHED", "新的训练目标没有超过 checkpoint 已完成步数。")

    runtime_description = {
        "python_version": runtime.python_version,
        "pytorch_version": runtime.pytorch_version,
    }
    runtime_description.update(request.get("_device_report", {}))
    environment_fingerprint_sha256 = request.get("_environment_fingerprint_sha256")
    if isinstance(environment_fingerprint_sha256, str):
        runtime_description["environment_fingerprint_sha256"] = environment_fingerprint_sha256
    signature = runtime.training_run_signature(request)
    migration_report = None
    if migration_source is not None:
        source_payload, source_manifest_record = runtime.load_verified_checkpoint_payload(
            migration_source
        )
        source_state = source_payload.get("model_state")
        if not isinstance(source_state, dict):
            raise UserError(
                "MIGRATION_SOURCE_MODEL_MISSING",
                "父 checkpoint 不包含可迁移的模型权重。",
            )
        source_digest = source_payload.get("model_state_sha256")
        if (
            not isinstance(source_digest, str)
            or runtime.state_dict_digest(source_state) != source_digest
        ):
            raise UserError(
                "CHECKPOINT_CORRUPT",
                "父 checkpoint 的模型状态与记录的 SHA-256 不一致。",
            )
        source_alias_groups = source_payload.get("model_state_alias_groups")
        if not isinstance(source_alias_groups, list):
            raise UserError(
                "MIGRATION_SHARED_IDENTITY_UNKNOWN",
                "父 checkpoint 未记录模型共享参数身份，无法安全推断结构迁移关系。",
                "请从支持共享身份记录的版本重新训练父运行，再执行权重迁移。",
            )
        migrated_state, plan_report = runtime.migrate_state_dict(
            source_state,
            model.state_dict(),
            migration_plan,
            source_alias_groups,
            runtime.model_state_alias_groups(model),
        )
        try:
            model.load_state_dict(migrated_state, strict=True)
        except Exception as error:
            raise UserError(
                "MIGRATION_TARGET_LOAD_FAILED",
                f"迁移后的状态无法载入目标模型：{type(error).__name__}: {error}",
                "迁移失败；父运行保持不变，子运行没有发布 checkpoint。",
            ) from error
        if runtime.model_state_digest(model) != runtime.state_dict_digest(migrated_state):
            raise UserError(
                "MIGRATION_TARGET_VERIFY_FAILED",
                "目标模型载入后的权重与迁移结果不一致。",
                "迁移失败；父运行保持不变，子运行没有发布 checkpoint。",
            )
        migration_report = {
            **plan_report,
            "parent_run_id": request.get("parent_run_id"),
            "source_checkpoint_sha256": source_manifest_record["sha256"],
            "source_completed_steps": source_payload["completed_steps"],
            "child_initial_step": 0,
            "source_model_state_sha256": source_digest,
            "child_initial_model_state_sha256": runtime.model_state_digest(model),
        }
        request["_migration_report"] = migration_report
        migrated_payload = {
            "schema_version": 1,
            "training_mode": mode,
            "completed_steps": 0,
            "run_signature": signature,
            "source_manifest": source_manifest,
            "source_snapshot_sha256": (request.get("code_snapshot") or {}).get("sha256"),
            "runtime": runtime_description,
            "environment_inventory": request.get("_checkpoint_environment_inventory"),
            "model_state": model.state_dict(),
            "model_state_sha256": runtime.model_state_digest(model),
            "model_state_alias_groups": runtime.model_state_alias_groups(model),
            "engine_state": {},
            "batch_state": None,
            "step_state": None,
            "loop_state": None,
            "rng_state": runtime.checkpoint_rng_state(),
            "weight_migration": migration_report,
        }
        check_snapshot_project_imports(request["project_root"], request.get("code_snapshot_root"),
                                       import_roots=request.get("runtime", {}).get("project_import_roots"))
        initial_manifest = runtime.save_training_checkpoint(directory, migrated_payload)
        request["_migration_initial_checkpoint_sha256"] = initial_manifest["sha256"]
    resume_state_status = (
        {"external_loop": loop_state_status}
        if mode == "external-loop"
        else {"batch": batch_state_status, "training_step": step_state_status}
    )
    checkpoint_state_status = (
        {"external_loop": "not-saved"}
        if mode == "external-loop"
        else {"batch": "not-saved", "training_step": "not-saved"}
    )

    def checkpoint(completed_steps, engine_state):
        stop_path = request.get("stop_file")
        stop_requested = False
        if stop_path:
            candidate = Path(stop_path)
            try:
                stop_requested = candidate.is_file() and not candidate.is_symlink()
            except OSError as error:
                raise UserError("STOP_FILE_UNAVAILABLE", f"无法读取停止请求：{error}") from error

        def emit_progress(checkpoint_saved):
            if not request.get("interactive_progress"):
                return
            metric_state = engine_state.get("metrics", {})
            if mode == "managed-step":
                loss_sum = metric_state.get("loss_sums", {}).get("loss", 0.0)
                loss_count = metric_state.get("loss_counts", {}).get("loss", 0)
            else:
                loss_sum = metric_state.get("loss_sum", 0.0)
                loss_count = metric_state.get("loss_count", 0)
            runtime.emit_progress({
                "stage": "training",
                "step": completed_steps,
                "total": steps,
                "mean_loss": loss_sum / loss_count if loss_count else None,
                "checkpoint": checkpoint_saved,
            })
        force_checkpoint = bool(request.pop("_force_checkpoint", False))
        if (
            not force_checkpoint
            and completed_steps % checkpoint_every != 0
            and completed_steps != steps
            and not stop_requested
        ):
            emit_progress(False)
            return False
        if mode == "external-loop":
            batch_state = None
            step_state = None
            loop_state = runtime.callable_checkpoint_hooks(loop_fn, "external-loop")
            if loop_state is not None and not runtime.checkpoint_value_supported(loop_state):
                raise UserError("CHECKPOINT_STATE_UNSUPPORTED", "external-loop 状态不是安全的 checkpoint 数据。")
            checkpoint_state_status["external_loop"] = (
                "saved"
                if isinstance(loop_state, dict) and loop_state.get("kind") in {"hooks", "versioned-hooks"}
                else "stateless"
                if isinstance(loop_state, dict) and loop_state.get("kind") == "stateless"
                else "reset-on-resume"
            )
        else:
            batch_state = runtime.callable_checkpoint_hooks(batch_fn, "batch 工厂")
            step_state = runtime.callable_checkpoint_hooks(step_fn, "training_step")
            loop_state = None
            if batch_state is not None and not runtime.checkpoint_value_supported(batch_state):
                raise UserError("CHECKPOINT_STATE_UNSUPPORTED", "batch 工厂状态不是安全的 checkpoint 数据。")
            if step_state is not None and not runtime.checkpoint_value_supported(step_state):
                raise UserError("CHECKPOINT_STATE_UNSUPPORTED", "training_step 状态不是安全的 checkpoint 数据。")
        if mode != "external-loop":
            checkpoint_state_status["batch"] = "saved" if batch_state else "reset-on-resume"
            checkpoint_state_status["training_step"] = "saved" if step_state else "reset-on-resume"
        payload = {
            "schema_version": 1,
            "training_mode": mode,
            "completed_steps": completed_steps,
            "run_signature": signature,
            "source_manifest": source_manifest,
            "source_snapshot_sha256": (request.get("code_snapshot") or {}).get("sha256"),
            "runtime": runtime_description,
            "environment_inventory": request.get("_checkpoint_environment_inventory"),
            "model_state": model.state_dict() if model is not None else None,
            "model_state_sha256": runtime.model_state_digest(model) if model is not None else None,
            "model_state_alias_groups": runtime.model_state_alias_groups(model) if model is not None else None,
            "weight_migration": request.get("_migration_report"),
            "engine_state": engine_state,
            "batch_state": batch_state,
            "step_state": step_state,
            "loop_state": loop_state,
            "rng_state": runtime.checkpoint_rng_state(),
        }
        check_snapshot_project_imports(request["project_root"], request.get("code_snapshot_root"),
                                       import_roots=request.get("runtime", {}).get("project_import_roots"))
        manifest = runtime.save_training_checkpoint(directory, payload)
        request["_latest_checkpoint"] = manifest
        emit_progress(True)
        return stop_requested

    if mode == "external-loop":
        result = runtime.run_external_loop(
            model,
            request,
            restore=restore,
            checkpoint_callback=checkpoint,
            loop_fn=loop_fn,
            loop_state_status=loop_state_status,
        )
    else:
        runner = runtime.run_managed_steps if mode == "managed-step" else runtime.run_manual_steps
        result = runner(model, request, restore=restore, checkpoint_callback=checkpoint)
    manifest = request.get("_latest_checkpoint")
    if not manifest:
        raise UserError("CHECKPOINT_MISSING", "训练完成但没有生成可验证的 checkpoint。")
    result.update({
        "checkpoint_saved": True,
        "checkpoint": manifest,
        "checkpoint_completed_steps": manifest["completed_steps"],
        "model_state_sha256": manifest["model_state_sha256"],
        "resumed_from_steps": start_step,
        "resume_state": resume_state_status,
        "checkpoint_state": checkpoint_state_status,
        "checkpoint_interval": checkpoint_every,
        "coverage": (
            (
                "正式 external-loop（未提供模型）；循环提供的 state_dict/load_state_dict 状态、用户上报步骤统计及 CPU/Python/CUDA RNG（设备可用时）原子保存。循环按报告边界响应协作停止；未声明状态时恢复需要显式 --allow-state-reset。"
                if model is None
                else "正式 external-loop；模型、循环提供的 state_dict/load_state_dict 状态、用户上报步骤统计及 CPU/Python/CUDA RNG（设备可用时）原子保存。循环按报告边界响应协作停止；未声明状态时恢复需要显式 --allow-state-reset。"
            )
            if mode == "external-loop"
            else "正式 managed/manual 训练；模型、具名优化器、步骤统计、CPU/Python/CUDA RNG（设备可用时）和已声明的数据/步骤状态原子保存。Linux 受监督后台运行支持附加、协作停止和最近 checkpoint 恢复。"
        ),
    })
    if migration_report is not None:
        result["weight_migration"] = migration_report
        result["migration_initial_checkpoint_sha256"] = request[
            "_migration_initial_checkpoint_sha256"
        ]
    if request.get("_resume_checkpoint") is not None:
        result["resume_batch_binding_evidence"] = (
            "legacy_checkpoint_unverified" if request.get("_legacy_batch_binding_unverified")
            else "checkpoint_signature_or_automatic"
        )
        result["resume_checkpoint"] = request["_resume_checkpoint"]
        result["resume_device_comparison"] = request.get("_resume_device_comparison", {
            "status": "unknown", "compared_fields": [],
            "note": "checkpoint 未提供可比较的设备观察。",
        })
    return result
