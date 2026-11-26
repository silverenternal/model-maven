"""Managed AdamW step execution with finite-value and gradient checks."""

import math
import sys
import time
import traceback

from ..protocol import UserError
from ..data.augment import apply_batch_augmentations
from .device import is_out_of_memory, out_of_memory_error
from .state import SampleIdentityTracker
from .optimizers import construct_optimizer


def run_managed_steps(runtime, model, request, restore=None, checkpoint_callback=None):
    steps = request.get("steps")
    max_steps = 1_000_000 if request.get("operation") == "train" else 1000
    if type(steps) is not int or not 1 <= steps <= max_steps:
        raise UserError("TRAINING_PARAMETER_INVALID", f"managed-step 的步数必须在 1..{max_steps}。")
    factory_ref = request.get("optimizer_factory")
    if factory_ref is None and request.get("optimizer", "adamw").lower() != "adamw":
        raise UserError("OPTIMIZER_UNSUPPORTED", "当前 managed-step 只支持 AdamW。")
    lr = request.get("lr")
    weight_decay = request.get("weight_decay")
    if factory_ref is None and (
        not isinstance(lr, (int, float))
        or isinstance(lr, bool)
        or not math.isfinite(lr)
        or lr <= 0
        or not isinstance(weight_decay, (int, float))
        or isinstance(weight_decay, bool)
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise UserError("TRAINING_PARAMETER_INVALID", "学习率/weight decay 数值无效。")
    if not request.get("batch"):
        raise UserError("BATCH_REQUIRED", "步骤试跑需要一个批次工厂。", "使用 --batch module:batch_factory。")
    if not request.get("training_step"):
        raise UserError("STEP_REQUIRED", "步骤试跑需要 training_step 入口。", "使用 --step module:training_step。")

    step_fn = runtime.entrypoint(request["training_step"])
    batch_fn = runtime.entrypoint(request["batch"])
    parameters = list(model.parameters())
    if not parameters:
        raise UserError("MODEL_PARAMETERS_MISSING", "模型没有可供优化器更新的参数。")
    if factory_ref is None:
        optimizer = runtime.torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))
        optimizer_identity = None
    else:
        optimizer, optimizer_identity = construct_optimizer(runtime, model, factory_ref, request.get("optimizer_params", {}))
    optimized_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    engine_restore = (restore or {}).get("engine_state", {})
    if engine_restore.get("optimizer_state") is not None and optimizer_identity is not None:
        if engine_restore.get("optimizer_identity") != optimizer_identity:
            raise UserError("RESUME_OPTIMIZER_INCOMPATIBLE", "优化器工厂、参数、类型或参数组与 checkpoint 不一致。")
    elif engine_restore.get("optimizer_state") is not None and engine_restore.get("optimizer_identity") is not None:
        raise UserError("RESUME_OPTIMIZER_INCOMPATIBLE", "自定义优化器 checkpoint 不能使用默认 AdamW 恢复。")
    if engine_restore.get("optimizer_state") is not None:
        try:
            optimizer.load_state_dict(engine_restore["optimizer_state"])
        except Exception as error:
            raise UserError(
                "RESUME_OPTIMIZER_INCOMPATIBLE",
                f"managed-step 优化器状态无法载入：{error}",
            ) from error
    model.train(True)
    metric_state = engine_restore.get("metrics", {})
    sums = metric_state.get("loss_sums", {"loss": 0.0})
    counts = metric_state.get("loss_counts", {"loss": 0})
    loss_sums = metric_state.get("named_loss_sums", {})
    loss_counts = metric_state.get("named_loss_counts", {})
    metric_sums = metric_state.get("metric_sums", {})
    metric_counts = metric_state.get("metric_counts", {})
    completed = int((restore or {}).get("completed_steps", 0))
    gradient_observations = int(engine_restore.get("gradient_observations", 0))
    sample_identity = SampleIdentityTracker(
        engine_restore.get("sample_identity"), prior_steps=completed
    )
    updated_names = set(engine_restore.get("updated_names", []))
    last_outputs = None
    last_loss = None
    started = time.perf_counter()
    supervision = request.get("_training_supervision")
    if supervision is not None:
        supervision.restore_coverage(
            engine_restore.get("supervision_coverage"), prior_steps=completed
        )

    for index in range(completed, steps):
        context = runtime.step_context(index + 1, steps)
        if supervision is not None:
            context._sample_ids_callback = supervision.report_sample_ids
        phase = "batch creation"
        try:
            batch = batch_fn(**request.get("batch_params", {}))
            if request.get("train_augment"):
                batch, augmentation_observations = apply_batch_augmentations(
                    batch,
                    request["train_augment"],
                    resolve_entrypoint=runtime.entrypoint,
                    seed=int(request.get("seed") or 0) + index,
                )
                request["_last_augmentation_observations"] = augmentation_observations
                request["_augmentation_batch_count"] = request.get("_augmentation_batch_count", 0) + 1
            phase = "batch device transfer"
            if runtime.move_batch is not None and request.get("batch_device_policy") != "preserve":
                if request.get("batch_device_policy") == "recursive-strict":
                    batch = runtime.move_batch(batch, request.get("_resolved_device", "cpu"), strict=True)
                else:
                    batch = runtime.move_batch(batch, request.get("_resolved_device", "cpu"))
            phase = "training step (forward/backward/update)"
            if optimizer_identity is not None:
                model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            if supervision is None:
                result = step_fn(model, batch, context, **request.get("training_params", {}))
            else:
                with supervision.capture_step(index + 1, batch=batch):
                    result = step_fn(model, batch, context, **request.get("training_params", {}))
            if not isinstance(result, dict):
                raise UserError("STEP_RESULT_INVALID", "training_step 必须返回包含 loss 的对象。")
            loss = result.get("loss")
            if not isinstance(loss, runtime.torch.Tensor) or loss.numel() != 1 or not loss.is_floating_point() or not loss.requires_grad:
                raise UserError("STEP_RESULT_INVALID", "training_step.loss 必须是依赖模型图的浮点标量 Tensor。")
            supervised_losses = {}
            if supervision is not None:
                loss, supervised_losses = supervision.complete_step(loss)
            if not bool(runtime.torch.isfinite(loss.detach()).all().item()):
                raise UserError("STEP_RESULT_INVALID", "training_step.loss 包含 NaN 或无穷大。")
            named_losses = runtime.scalar_map(result.get("losses"), "losses")
            if set(named_losses).intersection(supervised_losses):
                raise UserError(
                    "SUPERVISION_LOSS_NAME_CONFLICT",
                    "training_step 已使用保留的 supervision/<rule-id> loss 名称。",
                    "更改用户 loss 名称后重试；本步尚未更新参数。",
                )
            named_losses.update(supervised_losses)
            metrics = runtime.scalar_map(result.get("metrics"), "metrics")
            metrics.update(runtime.scalar_map(context._metrics, "context.metrics"))
            loss.backward()

            gradient_count = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                gradient = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
                if not bool(runtime.torch.isfinite(gradient).all().item()):
                    raise UserError("GRADIENT_NONFINITE", f"参数 {name} 的梯度包含 NaN 或无穷大。")
                if id(parameter) in optimized_ids:
                    gradient_count += 1
                    updated_names.add(name)
            if gradient_count == 0:
                raise UserError("GRADIENT_MISSING", "本步没有优化器管理的模型参数获得梯度。")
            optimizer.step()
            for name, parameter in model.named_parameters():
                if not bool(runtime.torch.isfinite(parameter.detach()).all().item()):
                    raise UserError("PARAMETER_NONFINITE", f"优化器更新后参数 {name} 包含 NaN 或无穷大。")
            for state in optimizer.state.values():
                for state_name, value in state.items():
                    if isinstance(value, runtime.torch.Tensor) and not bool(runtime.torch.isfinite(value).all().item()):
                        raise UserError("OPTIMIZER_STATE_NONFINITE", f"优化器状态 {state_name} 包含 NaN 或无穷大。")

            current_loss = runtime.scalar_value(loss, "loss")
            runtime.add_finite_total(sums, counts, "loss", current_loss, "losses")
            for name, value in named_losses.items():
                runtime.add_finite_total(loss_sums, loss_counts, name, value, "losses")
            for name, value in metrics.items():
                runtime.add_finite_total(metric_sums, metric_counts, name, value, "metrics")
            last_outputs = result.get("outputs")
            last_loss = current_loss
            gradient_observations += gradient_count
            sample_identity.record_step(context._sample_ids_report)
            completed += 1
            if checkpoint_callback is not None:
                should_stop = checkpoint_callback(
                    completed,
                    {
                        "optimizer_state": optimizer.state_dict(),
                        **({"optimizer_identity": optimizer_identity} if optimizer_identity is not None else {}),
                        "metrics": {
                            "loss_sums": sums,
                            "loss_counts": counts,
                            "named_loss_sums": loss_sums,
                            "named_loss_counts": loss_counts,
                            "metric_sums": metric_sums,
                            "metric_counts": metric_counts,
                        },
                        "gradient_observations": gradient_observations,
                        "updated_names": sorted(updated_names),
                        "sample_identity": sample_identity.checkpoint_state(),
                        "supervision_coverage": supervision.checkpoint_coverage()
                        if supervision is not None else None,
                    },
                )
                if should_stop:
                    raise UserError("USER_INTERRUPTED", "收到停止请求；已完成步骤并保存恢复点。")
        except UserError as error:
            if error.code == "USER_INTERRUPTED":
                raise
            raise UserError(
                "STEP_FAILED",
                f"第 {index + 1}/{steps} 步失败：{error}",
                (error.hint + " 正式训练保留最近一个完整 checkpoint（若已到保存间隔）。"
                 if error.code in {"BATCH_DEVICE_MAPPING_UNSUPPORTED", "BATCH_DEVICE_CONTAINER_UNSUPPORTED", "BATCH_DEVICE_CYCLE_UNSUPPORTED", "BATCH_DEVICE_STRUCTURE_LIMIT", "BATCH_DEVICE_SHARED_STORAGE_UNSUPPORTED"} and error.hint
                 else "正式训练保留最近一个完整 checkpoint（若已到保存间隔）；步骤试跑的权重仍为临时状态。"),
                {"failed_step": index + 1, "completed_steps": completed, "total_steps": steps,
                 "cause_code": error.code, "cause_message": str(error),
                 "cause_data": error.data, "cause_hint": error.hint, "phase": phase,
                 "training_step_started": phase == "training step (forward/backward/update)",
                 **({"training_mode": "managed-step", "managed_backward_started": False,
                     "managed_optimizer_step_started": False}
                    if error.code == "SUPERVISION_REQUIRE_FAILED" else {}),
                 "last_completed_loss": last_loss},
            ) from error
        except Exception as error:
            selection = request.get("_device_selection")
            if selection is not None and is_out_of_memory(runtime.torch, error):
                raise out_of_memory_error(runtime.torch, selection, phase, error) from error
            traceback.print_exc(file=sys.stderr)
            raise UserError(
                "STEP_FAILED",
                f"第 {index + 1}/{steps} 步失败：{type(error).__name__}: {error}",
                "正式训练保留最近一个完整 checkpoint（若已到保存间隔）；步骤试跑的权重仍为临时状态。",
                {"failed_step": index + 1, "completed_steps": completed, "total_steps": steps,
                 "cause_code": "USER_STEP_EXCEPTION", "cause_message": str(error),
                 "last_completed_loss": last_loss},
            ) from error

    elapsed = time.perf_counter() - started
    report = {
        "training_mode": "managed-step",
        "training_step": request["training_step"],
        "optimizer": optimizer_identity["optimizer_type"] if optimizer_identity else "AdamW",
        "learning_rate": float(lr) if optimizer_identity is None else None,
        "weight_decay": float(weight_decay) if optimizer_identity is None else None,
        **({"optimizer_identity": optimizer_identity} if optimizer_identity is not None else {}),
        "requested_steps": steps,
        "completed_steps": completed,
        "optimizer_updates": completed,
        "trainable_parameters": sum(1 for parameter in model.parameters() if parameter.requires_grad),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "parameters_with_gradient": len(updated_names),
        "gradient_parameters": sorted(updated_names)[:256],
        "gradient_parameters_truncated": len(updated_names) > 256,
        "gradient_observations": gradient_observations,
        "mean_loss": sums["loss"] / counts["loss"],
        "last_loss": last_loss,
        "mean_losses": {name: loss_sums[name] / loss_counts[name] for name in loss_sums},
        "mean_metrics": {name: metric_sums[name] / metric_counts[name] for name in metric_sums},
        "last_outputs": runtime.tree(last_outputs),
        "sample_identity": sample_identity.report(),
        "elapsed_seconds": elapsed,
        "checkpoint_saved": False,
        "coverage": "在独立 Python worker 进程中执行用户步骤、所选优化器更新和有限性检查；是否持久化由 try/train 操作决定。",
    }
    if supervision is not None:
        report["supervision"] = supervision.report()
        report["objective_applied"] = any(
            rule["mode"] == "penalty" for rule in report["supervision"]["rules"]
        )
    return report

