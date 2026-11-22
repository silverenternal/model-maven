"""User-owned full-loop progress, budget, cancellation and checkpoint context."""

import json
import math
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from ..protocol import InconclusiveError, UserError
from ..data.augment import apply_batch_augmentations
from ..run.artifacts import managed_artifact_declaration, run_artifact_directory
from .device import is_out_of_memory, out_of_memory_error
from .state import SampleIdentityTracker, sample_ids_report


class ExternalLoopContext:
    """Progress and cancellation port for a user-owned complete training loop."""

    def __init__(
        self,
        runtime,
        model,
        request,
        *,
        completed_steps=0,
        checkpoint_callback=None,
        resume_state=None,
        resume_state_status="new",
        engine_state=None,
        supervision=None,
        deadline_seconds=None,
        deadline_monotonic=None,
    ):
        self.runtime = runtime
        self.model = model
        self.total_steps = request["steps"]
        self.completed_steps = completed_steps
        self.resume_state = resume_state
        self.resume_state_status = resume_state_status
        self._request = request
        self._augmentation_calls_this_step = 0
        self._augmentation_observations = []
        if deadline_monotonic is not None and deadline_seconds is not None:
            raise UserError(
                "TRAINER_DEADLINE_INVALID",
                "只能提供一个 Trainer deadline 表示。",
            )
        if deadline_seconds is not None and (
            isinstance(deadline_seconds, bool)
            or not isinstance(deadline_seconds, (int, float))
            or not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
        ):
            raise UserError(
                "TRAINER_DEADLINE_INVALID",
                "Trainer deadline 必须是有限正数秒。",
            )
        if deadline_monotonic is not None and (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
        ):
            raise UserError("TRAINER_DEADLINE_INVALID", "Trainer deadline 必须是有限 monotonic timestamp。")
        self._deadline_monotonic = (
            float(deadline_monotonic)
            if deadline_monotonic is not None
            else time.monotonic() + float(deadline_seconds)
            if deadline_seconds is not None
            else None
        )
        self._timeout_reached = False
        self.device = str(request.get("_resolved_device", "cpu"))
        self.to_device = (
            (lambda batch: runtime.move_batch(batch, self.device))
            if runtime.move_batch is not None
            else None
        )
        self._checkpoint_callback = checkpoint_callback
        self._stop_file = request.get("stop_file")
        metrics = (engine_state or {}).get("metrics", {})
        self._loss_sum = float(metrics.get("loss_sum", 0.0))
        self._loss_count = int(metrics.get("loss_count", 0))
        self._metric_sums = dict(metrics.get("metric_sums", {}))
        self._metric_counts = dict(metrics.get("metric_counts", {}))
        self._last_loss = metrics.get("last_loss")
        self._pending_metrics = {}
        self._artifact_directory = run_artifact_directory(request.get("checkpoint_dir"))
        self._managed_artifacts = []
        engine_state = engine_state or {}
        self._sample_identity = SampleIdentityTracker(
            engine_state.get("sample_identity"), prior_steps=completed_steps
        )
        self._pending_sample_ids_report = None
        self.supervision = supervision
        self._supervised_capture = False
        self._supervised_step_ready = False
        self._supervised_updates = 0
        self._supervision_losses = {}
        self._optimizer_hooks = {}
        if self.supervision is not None:
            self.supervision.restore_coverage(
                engine_state.get("supervision_coverage") if isinstance(engine_state, dict) else None,
                prior_steps=completed_steps,
            )

    @property
    def remaining_steps(self):
        return max(0, self.total_steps - self.completed_steps)

    def transform_batch(self, batch):
        """Apply the project's explicit train augmentation targets to a user batch."""
        specifications = self._request.get("train_augment", [])
        call_index = self._request.get("_augmentation_batch_count", 0)
        transformed, observations = apply_batch_augmentations(
            batch,
            specifications,
            resolve_entrypoint=self.runtime.entrypoint,
            seed=int(self._request.get("seed") or 0) + call_index,
        )
        if specifications:
            self._augmentation_calls_this_step += 1
            self._augmentation_observations = observations
            self._request["_last_augmentation_observations"] = observations
            self._request["_augmentation_batch_count"] = self._request.get("_augmentation_batch_count", 0) + 1
        return transformed

    @property
    def remaining_time_seconds(self):
        """Seconds left in the runner-owned cooperative deadline, or None."""
        if self._deadline_monotonic is None:
            return None
        return max(0.0, self._deadline_monotonic - time.monotonic())

    @property
    def timeout_requested(self):
        """Whether the cooperative deadline expired; check before starting a step."""
        return self.remaining_time_seconds is not None and self.remaining_time_seconds <= 0.0

    @property
    def stop_requested(self):
        if not self._stop_file:
            return False
        candidate = Path(self._stop_file)
        try:
            return candidate.is_file() and not candidate.is_symlink()
        except OSError as error:
            raise UserError("STOP_FILE_UNAVAILABLE", f"无法读取停止请求：{error}") from error

    def log(self, name, value):
        if not isinstance(name, str) or not name or len(name) > 128:
            raise UserError("EXTERNAL_LOOP_EVENT_INVALID", "Metric 名称必须是 1..128 字符。")
        if len(self._pending_metrics) >= 128 and name not in self._pending_metrics:
            raise UserError("EXTERNAL_LOOP_EVENT_INVALID", "每步最多记录 128 个 Metric。")
        self._pending_metrics[name] = value

    def report_sample_ids(self, sample_ids, stable=True):
        """Report IDs used by the next completed update; raw IDs are discarded."""
        if self._pending_sample_ids_report is not None:
            raise UserError("EXTERNAL_LOOP_EVENT_INVALID", "每个步骤只能报告一次 sample IDs。")
        if self.supervision is not None:
            if not self._supervised_capture:
                raise UserError(
                    "EXTERNAL_SUPERVISION_STEP_REQUIRED",
                    "启用监督时，context.report_sample_ids() 必须在 context.supervised_step(batch) 内调用。",
                )
            self.supervision.report_sample_ids(sample_ids, stable=stable)
        self._pending_sample_ids_report = sample_ids_report(sample_ids, stable)

    @contextmanager
    def supervised_step(self, batch=None):
        """Capture manually marked ports for one user-owned external update."""
        if self.supervision is None:
            raise UserError(
                "EXTERNAL_SUPERVISION_NOT_ENABLED",
                "当前运行没有启用 --apply-supervision。",
            )
        if not self._optimizer_hooks:
            raise UserError(
                "EXTERNAL_SUPERVISION_OPTIMIZER_REQUIRED",
                "受监督 external-loop 步骤开始前，至少登记一个模型 optimizer。",
                "先调用 context.register_optimizer(optimizer)，再进入 context.supervised_step(batch)。",
            )
        if self._supervised_capture or self._supervised_step_ready:
            raise UserError(
                "EXTERNAL_SUPERVISION_STEP_STATE",
                "上一条受监督步骤尚未通过 context.report_step() 完成。",
            )
        self._supervised_capture = True
        self._supervised_updates = 0
        self._supervision_losses = {}
        try:
            with self.supervision.capture_step(self.completed_steps + 1, batch=batch):
                yield self
        finally:
            self._supervised_capture = False
        self._supervised_step_ready = True

    def backward(self, loss, retain_graph=False):
        """Add configured penalties and backpropagate one explicit external loss."""
        if self.supervision is None or not self._supervised_capture:
            raise UserError(
                "EXTERNAL_SUPERVISION_STEP_REQUIRED",
                "context.backward() 只能在启用监督的 context.supervised_step() 内调用。",
            )
        if (
            not isinstance(loss, self.runtime.torch.Tensor)
            or loss.numel() != 1
            or not loss.is_floating_point()
            or not loss.requires_grad
        ):
            raise UserError("LOSS_INVALID", "context.backward 需要依赖计算图的浮点标量 Tensor。")
        if not bool(self.runtime.torch.isfinite(loss.detach()).all().item()):
            raise UserError("LOSS_NONFINITE", "context.backward 收到的 Loss 含 NaN 或无穷大。")
        total_loss, self._supervision_losses = self.supervision.apply_manual_loss(loss)
        total_loss.backward(retain_graph=retain_graph)
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            if not bool(self.runtime.torch.isfinite(values).all().item()):
                raise UserError("GRADIENT_NONFINITE", f"参数 {name} 的梯度包含 NaN 或无穷大。")
        return total_loss

    def register_optimizer(self, optimizer):
        """Install pre/post-step guards on an optimizer used by this external loop."""
        if self.supervision is None:
            raise UserError(
                "EXTERNAL_SUPERVISION_NOT_ENABLED",
                "当前运行没有启用 --apply-supervision。",
            )
        identity = id(optimizer)
        if identity in self._optimizer_hooks:
            return optimizer
        pre_hook = getattr(optimizer, "register_step_pre_hook", None)
        post_hook = getattr(optimizer, "register_step_post_hook", None)
        if not callable(pre_hook) or not callable(post_hook):
            raise UserError(
                "EXTERNAL_SUPERVISION_OPTIMIZER_UNSUPPORTED",
                "该 optimizer 不提供可验证的 step 前后钩子，不能用于受监督 external-loop。",
                "使用支持 register_step_pre_hook/register_step_post_hook 的 PyTorch optimizer，或改用 manual-step。",
            )

        def before_step(current, args, kwargs):
            if not self._supervised_capture:
                raise UserError(
                    "EXTERNAL_SUPERVISION_STEP_REQUIRED",
                    "启用监督时，每次模型 optimizer.step() 都必须位于 context.supervised_step() 内。",
                )
            self.supervision.before_manual_update()
            for name, parameter in self.model.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                values = gradient.coalesce().values() if gradient.is_sparse else gradient
                if not bool(self.runtime.torch.isfinite(values).all().item()):
                    raise UserError("GRADIENT_NONFINITE", f"参数 {name} 的梯度包含 NaN 或无穷大。")
            self._supervised_updates += 1

        def after_step(current, args, kwargs):
            for name, parameter in self.model.named_parameters():
                if not bool(self.runtime.torch.isfinite(parameter.detach()).all().item()):
                    raise UserError("PARAMETER_NONFINITE", f"本步更新后参数 {name} 含 NaN 或无穷大。")

        self._optimizer_hooks[identity] = (pre_hook(before_step), post_hook(after_step))
        return optimizer

    def close(self):
        """Remove optimizer hooks so user-owned optimizers cannot retain this run context."""
        for hooks in self._optimizer_hooks.values():
            for handle in hooks:
                handle.remove()
        self._optimizer_hooks.clear()

    def report_step(self, loss, metrics=None):
        """Report one completed user-owned update; return False at the step budget."""
        if self._timeout_reached:
            raise UserError(
                "TRAINER_TIMEOUT_IGNORED",
                "Trainer 在 deadline 边界返回 False 后仍继续报告更新。",
                "检查 context.report_step() 的返回值，并在 context.timeout_requested 为真时停止启动新步骤。",
                {"completed_steps": self.completed_steps, "total_steps": self.total_steps},
            )
        if self.completed_steps >= self.total_steps:
            raise UserError(
                "TRAINING_BUDGET_EXCEEDED",
                f"external-loop 已达到步骤预算 {self.total_steps}。",
                "检查 report_step() 的返回值，达到预算时结束用户循环。",
            )
        if self._request.get("train_augment") and self._augmentation_calls_this_step == 0:
            raise UserError(
                "EXTERNAL_AUGMENT_NOT_APPLIED",
                "项目声明了训练增强，但该 external-loop 步骤没有调用 context.transform_batch()。",
                "对当前训练 batch 调用 context.transform_batch(batch)，再使用返回值更新模型。",
                {"completed_steps": self.completed_steps, "total_steps": self.total_steps},
            )
        loss_value = self.runtime.scalar_value(loss, "loss")
        reported_metrics = dict(self._pending_metrics)
        reported_metrics.update(self.runtime.scalar_map(metrics, "metrics"))
        if self.supervision is not None:
            if self._supervised_capture or not self._supervised_step_ready:
                raise UserError(
                    "EXTERNAL_SUPERVISION_STEP_INCOMPLETE",
                    "启用监督时，report_step() 必须在已退出 supervised_step 并完成 optimizer update 后调用。",
                )
            if self._supervised_updates < 1:
                raise UserError(
                    "EXTERNAL_SUPERVISION_UPDATE_MISSING",
                    "该 external-loop 步骤没有通过受监督 optimizer 完成更新。",
                    data={
                        "completed_steps": self.completed_steps,
                        "optimizer_updates": self._supervised_updates,
                        "supervised_step_ready": self._supervised_step_ready,
                        "registered_optimizers": len(self._optimizer_hooks),
                    },
                )
            supervision_losses = self.supervision.finish_manual_step()
            if set(reported_metrics).intersection(supervision_losses):
                raise UserError("SUPERVISION_LOSS_NAME_CONFLICT", "training loop 使用了保留的 supervision/<rule-id> 指标名称。")
            reported_metrics.update(supervision_losses)
        self._pending_metrics.clear()
        if self.model is not None:
            for parameter_name, parameter in self.model.named_parameters():
                if not bool(self.runtime.torch.isfinite(parameter.detach()).all().item()):
                    raise UserError(
                        "PARAMETER_NONFINITE",
                        f"external-loop 第 {self.completed_steps + 1} 步后参数 {parameter_name} 含 NaN 或无穷大。",
                    )
        next_loss_sum = self._loss_sum + loss_value
        if not math.isfinite(next_loss_sum):
            raise UserError("EXTERNAL_LOOP_EVENT_INVALID", "Loss 累计后超出有限数值范围。")
        self._loss_sum = next_loss_sum
        self._loss_count += 1
        self._last_loss = loss_value
        self._sample_identity.record_step(self._pending_sample_ids_report)
        self._pending_sample_ids_report = None
        for name, value in reported_metrics.items():
            self.runtime.add_finite_total(self._metric_sums, self._metric_counts, name, value, "metrics")
        self.completed_steps += 1
        self._augmentation_calls_this_step = 0
        self._supervised_step_ready = False
        self._supervised_updates = 0
        if self.timeout_requested:
            self._timeout_reached = True
            self._request["_force_checkpoint"] = True
        should_stop = False
        if self._checkpoint_callback is not None:
            should_stop = self._checkpoint_callback(self.completed_steps, self.engine_state())
        if should_stop:
            raise UserError(
                "USER_INTERRUPTED",
                "收到停止请求；external-loop 已在报告的步骤边界保存恢复点。",
                "最近 checkpoint 可通过 m-model train --resume 恢复。",
                {"completed_steps": self.completed_steps, "total_steps": self.total_steps},
            )
        return self.remaining_steps > 0 and not self._timeout_reached

    def engine_state(self):
        return {
            "sample_identity": self._sample_identity.checkpoint_state(),
            "supervision_coverage": self.supervision.checkpoint_coverage()
            if self.supervision is not None else None,
            "metrics": {
                "loss_sum": self._loss_sum,
                "loss_count": self._loss_count,
                "last_loss": self._last_loss,
                "metric_sums": self._metric_sums,
                "metric_counts": self._metric_counts,
            }
        }

    @property
    def artifact_dir(self):
        """Run-private directory for files explicitly registered as managed artifacts."""
        if self._artifact_directory is None:
            raise UserError(
                "RUN_ARTIFACTS_UNAVAILABLE",
                "当前操作没有可持久化的运行产物目录。",
                "仅正式训练运行可登记 managed artifacts。",
            )
        return self._artifact_directory

    def register_artifact(
        self,
        artifact_id,
        files,
        *,
        owner="external_loop",
        references=(),
        consumes=(),
        complete=True,
        kind="checkpoint",
    ):
        """Register files beneath artifact_dir; Rust rechecks paths, ownership data and hashes."""
        if len(self._managed_artifacts) >= 64:
            raise UserError("RUN_ARTIFACT_LIMIT", "每次训练最多登记 64 个 managed artifacts。")
        declaration = managed_artifact_declaration(
            self.artifact_dir,
            artifact_id,
            files,
            owner,
            references,
            consumes,
            complete,
            kind,
            self._request.get("run_id"),
        )
        if any(item["id"] == declaration["id"] for item in self._managed_artifacts):
            raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "managed artifact id 不能重复。")
        self._managed_artifacts.append(declaration)


def run_external_loop(
    runtime,
    model,
    request,
    *,
    restore=None,
    checkpoint_callback=None,
    loop_fn=None,
    loop_state_status="new",
):
    maximum = 1_000_000 if request.get("operation") == "train" else 1000
    steps = request.get("steps")
    if type(steps) is not int or not 1 <= steps <= maximum:
        raise UserError("TRAINING_PARAMETER_INVALID", f"external-loop 的步骤预算必须在 1..{maximum}。")
    loop_ref = request.get("training_loop")
    if not loop_ref:
        raise UserError("EXTERNAL_LOOP_REQUIRED", "external-loop 缺少用户循环入口。", "使用 --loop module:run_loop。")
    if loop_fn is None:
        loop_fn = runtime.entrypoint(loop_ref)
    completed = int((restore or {}).get("completed_steps", 0))
    runtime_options = request.get("runtime", {})
    deadline_seconds = (
        runtime_options.get("timeout_seconds")
        if isinstance(runtime_options, dict)
        else None
    )
    deadline_monotonic = request.get("_worker_deadline_monotonic")
    if deadline_monotonic is not None:
        deadline_seconds = None
    context = ExternalLoopContext(
        runtime,
        model,
        request,
        completed_steps=completed,
        checkpoint_callback=checkpoint_callback,
        resume_state=(restore or {}).get("loop_state", {}).get("state")
        if isinstance((restore or {}).get("loop_state"), dict)
        and (restore or {}).get("loop_state", {}).get("kind") == "hooks"
        else None,
        resume_state_status=loop_state_status,
        engine_state=(restore or {}).get("engine_state"),
        supervision=request.get("_training_supervision"),
        deadline_seconds=deadline_seconds,
        deadline_monotonic=deadline_monotonic,
    )
    if model is not None:
        model.train(True)
    started = time.perf_counter()
    try:
        loop_result = loop_fn(model, context, **request.get("training_params", {}))
    except UserError:
        raise
    except Exception as error:
        selection = request.get("_device_selection")
        if selection is not None and is_out_of_memory(runtime.torch, error):
            raise out_of_memory_error(runtime.torch, selection, "external-loop", error) from error
        traceback.print_exc(file=sys.stderr)
        raise UserError(
            "EXTERNAL_LOOP_FAILED",
            f"用户 external-loop 执行失败：{type(error).__name__}: {error}",
            "检查循环入口和 context.report_step() 调用；停止请求仅在用户轮询或报告步骤时协作生效。",
            {"completed_steps": context.completed_steps, "total_steps": steps},
        ) from error
    finally:
        context.close()
    if context.timeout_requested:
        context._timeout_reached = True
    if context.completed_steps == completed:
        if context.stop_requested:
            raise UserError(
                "USER_INTERRUPTED",
                "收到停止请求；循环尚未报告新步骤，保留已有 checkpoint（若有）。",
                "再次训练可从最近一次已提交 checkpoint 恢复。",
                {"completed_steps": context.completed_steps, "total_steps": steps},
            )
        error = InconclusiveError if request.get("operation") == "training_probe" else UserError
        raise error(
            "EXTERNAL_LOOP_NO_STEPS",
            "external-loop 返回前没有报告任何已完成步骤。",
            "每次完成一次更新后调用 context.report_step(loss, metrics)。未报告的循环不会标为训练成功。",
            {"training_mode": "external-loop", "completed_steps": 0, "total_steps": steps},
        )
    manifest = request.get("_latest_checkpoint")
    if checkpoint_callback is not None and (
        not manifest or manifest.get("completed_steps", -1) < context.completed_steps
    ):
        request["_force_checkpoint"] = True
        if checkpoint_callback(context.completed_steps, context.engine_state()):
            raise UserError(
                "USER_INTERRUPTED",
                "收到停止请求；external-loop 已在循环返回边界保存恢复点。",
                "最近 checkpoint 可通过 m-model train --resume 恢复。",
                {"completed_steps": context.completed_steps, "total_steps": steps},
            )
        manifest = request.get("_latest_checkpoint")
    context._elapsed_seconds = time.perf_counter() - started
    result_summary = None
    if loop_result is not None:
        try:
            result_summary = json.loads(json.dumps(loop_result, allow_nan=False))
        except (TypeError, ValueError):
            result_summary = {"type": f"{type(loop_result).__module__}.{type(loop_result).__qualname__}"}
    metric_state = context.engine_state()["metrics"]
    if callable(getattr(loop_fn, "state_dict", None)) and callable(
        getattr(loop_fn, "load_state_dict", None)
    ):
        loop_checkpoint_state = "saved"
    elif getattr(loop_fn, "MODEL_MAVEN_STATELESS", False) is True:
        loop_checkpoint_state = "stateless"
    elif manifest is not None:
        loop_checkpoint_state = "reset-on-resume"
    else:
        loop_checkpoint_state = "not-saved"
    return {
        "training_mode": "external-loop",
        "training_loop": loop_ref,
        "requested_steps": context.total_steps,
        "completed_steps": context.completed_steps,
        "stopped_early": context.completed_steps < context.total_steps,
        "termination_reason": (
            "timeout"
            if context._timeout_reached
            else "step_budget"
            if context.completed_steps >= context.total_steps
            else "trainer_returned"
        ),
        "mean_loss": context._loss_sum / context._loss_count,
        "last_loss": context._last_loss,
        "sample_identity": context._sample_identity.report(),
        "supervision": context.supervision.report() if context.supervision is not None else None,
        "mean_metrics": {
            name: total / metric_state["metric_counts"][name]
            for name, total in metric_state["metric_sums"].items()
        },
        "loop_result": result_summary,
        "managed_artifacts": context._managed_artifacts,
        "elapsed_seconds": context._elapsed_seconds,
        "checkpoint_saved": manifest is not None,
        "resume_state_status": context.resume_state_status,
        "capabilities": {
            "loop_progress": "reported",
            "parameter_finiteness": "checked_at_reported_steps" if model is not None else "not_applicable_no_model",
            "checkpoint_model": (
                "not_applicable_no_model"
                if model is None
                else "saved"
                if manifest is not None
                else "not_saved"
            ),
            "checkpoint_loop_state": loop_checkpoint_state,
            "forward_contract": "user_owned_not_independently_verified",
            "loss_contract": "user_reported_not_independently_verified",
            "backward_and_optimizer_updates": "explicit_external_supervision_hooks"
            if context.supervision is not None else "not_independently_verified",
            "resume_cursor": (
                "restorable" if loop_checkpoint_state in {"saved", "stateless"}
                else "explicit_reset_required" if loop_checkpoint_state == "reset-on-resume"
                else "unsupported"
            ),
        },
        "coverage": "完整训练循环由用户代码控制；启用监督时，用户通过 supervised_step、context.backward 和已登记 optimizer hooks 显式接入人工标记规则。否则 CLI 只负责进程、步骤预算、有限性抽查、协作停止和可选循环状态恢复，不独立验证 forward/backward/optimizer 逻辑。",
    }


