"""User-owned optimization steps with observable optimizer updates."""

import math
import re
import sys
import time
import traceback

from ..protocol import UserError
from ..data.augment import apply_batch_augmentations
from .device import is_out_of_memory, out_of_memory_error
from .state import SampleIdentityTracker


class TrackedOptimizer:
    """Narrow optimizer wrapper so manual steps remain observable and checkable."""

    def __init__(self, training_context, name, optimizer):
        self._context = training_context
        self.name = name
        self._optimizer = optimizer

    def zero_grad(self, set_to_none=True):
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self._context.check_optimizer_gradients(self.name, self._optimizer)
        result = self._optimizer.step(closure)
        self._context.updates[self.name] = self._context.updates.get(self.name, 0) + 1
        self._context.check_optimizer_values(self.name, self._optimizer)
        return result

    def __getattr__(self, name):
        return getattr(self._optimizer, name)


class ManualTrainingContext:
    """Per-run optimizer registry with a per-step view for user-owned update logic."""

    def __init__(self, runtime, model, request, restore=None):
        self.runtime = runtime
        self.model = model
        self.default_lr = float(request["lr"])
        self.default_weight_decay = float(request["weight_decay"])
        self.optimizers = {}
        self.optimizer_signatures = {}
        self.optimizer_metadata = {}
        self.parameter_owners = {}
        self.parameter_names = {}
        for parameter_name, parameter in model.named_parameters(remove_duplicate=False):
            self.parameter_names.setdefault(id(parameter), parameter_name)
        self.restore_optimizers = dict((restore or {}).get("optimizers", {}))
        self.updates = dict((restore or {}).get("updates", {}))
        self.gradient_names = set((restore or {}).get("gradient_names", []))
        self.current_step = 0
        self.supervision = request.get("_training_supervision")
        self.supervision_loss = None
        self.supervision_named_losses = {}

    def for_step(self, step, total_steps):
        self.current_step = step
        self.supervision_loss = None
        self.supervision_named_losses = {}
        context = self.runtime.step_context(step, total_steps)
        if self.supervision is not None:
            context._sample_ids_callback = self.supervision.report_sample_ids
        context.optimizer = self.register_optimizer
        context.backward = self.backward
        return context

    def register_optimizer(self, name, parameters, optimizer="adamw", lr=None, weight_decay=None):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", name):
            raise UserError("OPTIMIZER_REGISTRATION_INVALID", "优化器名称须为最多 64 位标识符。")
        if not isinstance(optimizer, str) or optimizer.lower() != "adamw":
            raise UserError("OPTIMIZER_UNSUPPORTED", "当前 manual-step 只支持注册 AdamW 优化器组。")
        try:
            parameters = list(parameters)
        except TypeError as error:
            raise UserError("OPTIMIZER_REGISTRATION_INVALID", f"优化器 {name} 的参数不可迭代。") from error
        if not parameters or any(not isinstance(parameter, self.runtime.torch.nn.Parameter) for parameter in parameters):
            raise UserError("OPTIMIZER_REGISTRATION_INVALID", f"优化器 {name} 必须包含模型 Parameter。")
        identities = tuple(id(parameter) for parameter in parameters)
        if len(set(identities)) != len(identities):
            raise UserError("OPTIMIZER_PARAMETER_DUPLICATE", f"优化器 {name} 重复登记了同一参数。")
        lr = self.default_lr if lr is None else lr
        weight_decay = self.default_weight_decay if weight_decay is None else weight_decay
        if (
            isinstance(lr, bool) or not isinstance(lr, (int, float))
            or not math.isfinite(lr) or lr <= 0
            or isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float))
            or not math.isfinite(weight_decay) or weight_decay < 0
        ):
            raise UserError("OPTIMIZER_REGISTRATION_INVALID", f"优化器 {name} 的学习率或 weight decay 无效。")
        signature = (identities, "adamw", float(lr), float(weight_decay))
        if name in self.optimizers:
            if self.optimizer_signatures[name] != signature:
                raise UserError("OPTIMIZER_REGISTRATION_CHANGED", f"优化器 {name} 在运行中改变了参数组或配置。")
            return self.optimizers[name]

        for parameter in parameters:
            owner = self.parameter_owners.get(id(parameter))
            if owner is not None:
                raise UserError(
                    "OPTIMIZER_PARAMETER_OVERLAP",
                    f"参数同时属于优化器 {owner} 和 {name}。",
                    "共享参数默认只登记到一个优化器；如需特殊共享更新策略，应在同一用户步骤中显式处理。",
                )
        optimizer_instance = self.runtime.torch.optim.AdamW(
            parameters, lr=float(lr), weight_decay=float(weight_decay)
        )
        parameter_names = [self.parameter_names.get(identity) for identity in identities]
        if any(name is None for name in parameter_names):
            raise UserError(
                "OPTIMIZER_PARAMETER_INVALID",
                f"优化器 {name} 包含不属于当前模型的 Parameter。",
            )
        metadata = {
            "parameter_names": parameter_names,
            "optimizer": "adamw",
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        }
        saved = self.restore_optimizers.pop(name, None)
        if saved is not None:
            if saved.get("metadata") != metadata:
                raise UserError(
                    "RESUME_OPTIMIZER_INCOMPATIBLE",
                    f"优化器组 {name} 的参数名称或配置与 checkpoint 不一致。",
                )
            try:
                optimizer_instance.load_state_dict(saved["state"])
            except Exception as error:
                raise UserError(
                    "RESUME_OPTIMIZER_INCOMPATIBLE",
                    f"优化器组 {name} 的状态无法载入：{error}",
                ) from error
            self.updates[name] = int(saved.get("updates", 0))
        else:
            self.updates.setdefault(name, 0)
        tracked = TrackedOptimizer(self, name, optimizer_instance)
        self.optimizers[name] = tracked
        self.optimizer_signatures[name] = signature
        self.optimizer_metadata[name] = metadata
        for parameter in parameters:
            self.parameter_owners[id(parameter)] = name
        return tracked

    def checkpoint_state(self):
        optimizers = dict(self.restore_optimizers)
        for name, optimizer in self.optimizers.items():
            optimizers[name] = {
                "metadata": self.optimizer_metadata[name],
                "state": optimizer._optimizer.state_dict(),
                "updates": self.updates.get(name, 0),
            }
        return {
            "optimizers": optimizers,
            "updates": dict(self.updates),
            "gradient_names": sorted(self.gradient_names),
        }

    def backward(self, loss, retain_graph=False):
        if not isinstance(loss, self.runtime.torch.Tensor) or loss.numel() != 1 or not loss.is_floating_point() or not loss.requires_grad:
            raise UserError("LOSS_INVALID", "context.backward 需要依赖计算图的浮点标量 Tensor。")
        if not bool(self.runtime.torch.isfinite(loss.detach()).all().item()):
            raise UserError("LOSS_NONFINITE", "context.backward 收到的 Loss 含 NaN 或无穷大。")
        if self.supervision is not None:
            loss, self.supervision_named_losses = self.supervision.apply_manual_loss(loss)
            self.supervision_loss = loss
        loss.backward(retain_graph=retain_graph)
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            if not bool(self.runtime.torch.isfinite(values).all().item()):
                raise UserError("GRADIENT_NONFINITE", f"参数 {name} 的梯度包含 NaN 或无穷大。")
            self.gradient_names.add(name)

    def check_optimizer_gradients(self, name, optimizer):
        if self.supervision is not None:
            self.supervision.before_manual_update()
        gradient_count = 0
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                values = gradient.coalesce().values() if gradient.is_sparse else gradient
                if not bool(self.runtime.torch.isfinite(values).all().item()):
                    raise UserError("GRADIENT_NONFINITE", f"优化器 {name} 收到非有限梯度。")
                gradient_count += 1
        if gradient_count == 0:
            raise UserError("GRADIENT_MISSING", f"优化器 {name} 没有可更新的梯度。")
        for parameter_name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                self.gradient_names.add(parameter_name)

    def check_optimizer_values(self, name, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if not bool(self.runtime.torch.isfinite(parameter.detach()).all().item()):
                    raise UserError("PARAMETER_NONFINITE", f"优化器 {name} 更新后产生非有限参数。")
        for state in optimizer.state.values():
            for state_name, value in state.items():
                if isinstance(value, self.runtime.torch.Tensor) and not bool(self.runtime.torch.isfinite(value).all().item()):
                    raise UserError("OPTIMIZER_STATE_NONFINITE", f"优化器 {name} 状态 {state_name} 含 NaN 或无穷大。")


def run_manual_steps(runtime, model, request, restore=None, checkpoint_callback=None):
    steps = request.get("steps")
    max_steps = 1_000_000 if request.get("operation") == "train" else 1000
    if type(steps) is not int or not 1 <= steps <= max_steps:
        raise UserError("TRAINING_PARAMETER_INVALID", f"manual-step 的步数必须在 1..{max_steps}。")
    if not request.get("batch"):
        raise UserError("BATCH_REQUIRED", "手动训练试跑需要批次工厂。", "使用 --batch module:batch_factory。")
    step_ref = request.get("training_step")
    if not step_ref:
        raise UserError("STEP_REQUIRED", "手动训练试跑需要 training_step 入口。", "使用 --step module:training_step。")
    step_fn = runtime.entrypoint(step_ref)
    batch_fn = runtime.entrypoint(request["batch"])
    engine_restore = (restore or {}).get("engine_state", {})
    context = ManualTrainingContext(runtime, model, request, engine_restore.get("manual_context"))
    model.train(True)
    metric_state = engine_restore.get("metrics", {})
    loss_sums = metric_state.get("named_loss_sums", {})
    loss_counts = metric_state.get("named_loss_counts", {})
    metric_sums = metric_state.get("metric_sums", {})
    metric_counts = metric_state.get("metric_counts", {})
    loss_sum = float(metric_state.get("loss_sum", 0.0))
    loss_count = int(metric_state.get("loss_count", 0))
    completed = int((restore or {}).get("completed_steps", 0))
    sample_identity = SampleIdentityTracker(
        engine_restore.get("sample_identity"), prior_steps=completed
    )
    if context.supervision is not None:
        context.supervision.restore_coverage(
            engine_restore.get("supervision_coverage"), prior_steps=completed
        )
    last_outputs = None
    last_loss = None
    started = time.perf_counter()

    for index in range(completed, steps):
        step_context = context.for_step(index + 1, steps)
        batch = None
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
            if runtime.move_batch is not None:
                batch = runtime.move_batch(batch, request.get("_resolved_device", "cpu"))
            phase = "training step (forward/backward/update)"
            updates_before = sum(context.updates.values())
            if context.supervision is None:
                result = step_fn(model, batch, step_context, **request.get("training_params", {}))
            else:
                with context.supervision.capture_step(index + 1, batch=batch):
                    result = step_fn(model, batch, step_context, **request.get("training_params", {}))
            if not isinstance(result, dict) or "loss" not in result:
                raise UserError("STEP_RESULT_INVALID", "manual-step 必须返回包含 loss 的对象。")
            result_loss = context.supervision_loss if context.supervision_loss is not None else result["loss"]
            current_loss = runtime.scalar_value(result_loss, "loss")
            named_losses = runtime.scalar_map(result.get("losses"), "losses")
            if set(named_losses).intersection(context.supervision_named_losses):
                raise UserError(
                    "SUPERVISION_LOSS_NAME_CONFLICT",
                    "training_step 已使用保留的 supervision/<rule-id> loss 名称。",
                )
            named_losses.update(context.supervision_named_losses)
            metrics = runtime.scalar_map(result.get("metrics"), "metrics")
            metrics.update(runtime.scalar_map(step_context._metrics, "context.metrics"))
            updates_this_step = sum(context.updates.values()) - updates_before
            if updates_this_step == 0:
                raise UserError("MANUAL_STEP_NO_UPDATE", "本步没有调用已注册优化器的 step()。")
            if context.supervision is not None:
                context.supervision.finish_manual_step()
            for parameter_name, parameter in model.named_parameters():
                if not bool(runtime.torch.isfinite(parameter.detach()).all().item()):
                    raise UserError("PARAMETER_NONFINITE", f"本步结束时参数 {parameter_name} 含 NaN 或无穷大。")
            next_loss_sum = loss_sum + current_loss
            if not math.isfinite(next_loss_sum):
                raise UserError("STEP_RESULT_INVALID", "Loss 累计后超出有限数值范围。")
            loss_sum, loss_count = next_loss_sum, loss_count + 1
            for name, value in named_losses.items():
                runtime.add_finite_total(loss_sums, loss_counts, name, value, "losses")
            for name, value in metrics.items():
                runtime.add_finite_total(metric_sums, metric_counts, name, value, "metrics")
            last_outputs = result.get("outputs")
            last_loss = current_loss
            sample_identity.record_step(step_context._sample_ids_report)
            completed += 1
            if checkpoint_callback is not None:
                should_stop = checkpoint_callback(
                    completed,
                    {
                        "sample_identity": sample_identity.checkpoint_state(),
                        "supervision_coverage": context.supervision.checkpoint_coverage()
                        if context.supervision is not None else None,
                        "manual_context": context.checkpoint_state(),
                        "metrics": {
                            "loss_sum": loss_sum,
                            "loss_count": loss_count,
                            "named_loss_sums": loss_sums,
                            "named_loss_counts": loss_counts,
                            "metric_sums": metric_sums,
                            "metric_counts": metric_counts,
                        },
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
                "manual-step 由用户代码管理反向和更新；正式训练保留最近一个完整 checkpoint（若已到保存间隔）。",
                {"failed_step": index + 1, "completed_steps": completed, "total_steps": steps,
                 "cause_code": error.code, "cause_message": str(error), "cause_data": error.data,
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
                "检查用户步骤与已注册优化器；正式训练保留最近一个完整 checkpoint（若已到保存间隔）。",
                {"failed_step": index + 1, "completed_steps": completed, "total_steps": steps,
                 "cause_code": "USER_STEP_EXCEPTION", "cause_message": str(error),
                 "last_completed_loss": last_loss},
            ) from error

    report = {
        "training_mode": "manual-step",
        "training_step": step_ref,
        "optimizer": "user-registered AdamW groups",
        "optimizer_groups": list(context.optimizers),
        "optimizer_updates_by_group": context.updates,
        "optimizer_updates": sum(context.updates.values()),
        "requested_steps": steps,
        "completed_steps": completed,
        "trainable_parameters": sum(1 for parameter in model.parameters() if parameter.requires_grad),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "parameters_with_gradient": len(context.gradient_names),
        "gradient_parameters": sorted(context.gradient_names)[:256],
        "gradient_parameters_truncated": len(context.gradient_names) > 256,
        "mean_loss": loss_sum / loss_count,
        "last_loss": last_loss,
        "mean_losses": {name: loss_sums[name] / loss_counts[name] for name in loss_sums},
        "mean_metrics": {name: metric_sums[name] / metric_counts[name] for name in metric_sums},
        "last_outputs": runtime.tree(last_outputs),
        "sample_identity": sample_identity.report(),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_saved": False,
        "coverage": "manual-step 用户代码负责反向、阶段和已注册优化器更新；正式训练按具名优化器组保存状态。",
    }
    if context.supervision is not None:
        report["supervision"] = context.supervision.report()
        report["objective_applied"] = any(
            rule["mode"] == "penalty" for rule in report["supervision"]["rules"]
        )
    return report


