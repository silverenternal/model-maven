"""Framework-neutral user loop primitives. No ML library is imported here.

Trials do not persist state. Formal runs save only the JSON state that user
code explicitly reports; external data and random state remain user-managed.
"""

import json
import math
import time
from pathlib import Path

from ..protocol import UserError

_MAX_STATE_BYTES = 8 * 1024 * 1024
_MAX_METRICS = 128


def _finite_metrics(metrics):
    if not isinstance(metrics, dict) or len(metrics) > _MAX_METRICS:
        raise UserError("GENERIC_LOOP_METRICS_INVALID", "指标必须是最多 128 项的对象。")
    result = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name or len(name) > 128:
            raise UserError("GENERIC_LOOP_METRICS_INVALID", "指标名称必须为 1 到 128 个字符。")
        try:
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise UserError("GENERIC_LOOP_METRICS_INVALID", f"指标 {name!r} 必须是有限数字。")
        result[name] = value
    return result


def _json_state(state):
    try:
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise UserError("GENERIC_LOOP_STATE_INVALID", "恢复状态必须是有限 JSON 数据。") from error
    if len(encoded) > _MAX_STATE_BYTES:
        raise UserError("GENERIC_LOOP_STATE_INVALID", "恢复状态不能超过 8 MiB。")
    # A copy prevents later user mutation from changing the reported state.
    return json.loads(encoded)


class GenericLoopContext:
    """Progress and optional state boundary; user code owns all computation."""

    def __init__(self, total_steps, *, completed_steps=0, resume_state=None,
                 stop_file=None, deadline_monotonic=None, report_callback=None):
        if type(total_steps) is not int or not 1 <= total_steps <= 1_000_000:
            raise UserError("TRAINING_PARAMETER_INVALID", "循环总步数必须在 1 到 1000000 之间。")
        if type(completed_steps) is not int or not 0 <= completed_steps < total_steps:
            raise UserError("GENERIC_LOOP_RESUME_INVALID", "已完成步数必须小于当前目标步数。")
        if deadline_monotonic is not None and (
            isinstance(deadline_monotonic, bool) or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
        ):
            raise UserError("TRAINER_DEADLINE_INVALID", "截止时间必须为有限 monotonic 时间。")
        self.total_steps = total_steps
        self.completed_steps = completed_steps
        self.resume_state = _json_state(resume_state) if resume_state is not None else None
        self._stop_file = Path(stop_file) if stop_file else None
        self._deadline = deadline_monotonic
        self._report_callback = report_callback
        self._last_metrics = None
        self._reports = 0
        self._closed = False

    @property
    def remaining_steps(self):
        return max(0, self.total_steps - self.completed_steps)

    @property
    def stop_requested(self):
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return True
        if self._stop_file is None:
            return False
        try:
            return self._stop_file.is_file() and not self._stop_file.is_symlink()
        except OSError as error:
            raise UserError("STOP_FILE_UNAVAILABLE", f"无法读取停止请求：{error}") from error

    def report_step(self, metrics, *, state=None):
        """Report one user-completed step; optional JSON state is user-owned.

        Omitting state or passing None does not request a checkpoint; wrap a
        JSON null in a container if it must be restored. Callback persistence
        failure aborts the report rather than claiming a durable step. Stop is
        cooperative: the caller must honor False.
        """
        if self._closed or self.completed_steps >= self.total_steps:
            raise UserError("TRAINING_BUDGET_EXCEEDED", "循环已结束或达到步骤预算。")
        if self.stop_requested:
            raise UserError("USER_INTERRUPTED", "已请求停止；没有接收新的步骤报告。",
                            data={"completed_steps": self.completed_steps})
        checked_metrics = _finite_metrics(metrics)
        checked_state = _json_state(state) if state is not None else None
        next_step = self.completed_steps + 1
        if self._report_callback is not None:
            self._report_callback(next_step, checked_metrics, checked_state)
        self.completed_steps = next_step
        self._reports += 1
        self._last_metrics = checked_metrics
        return self.remaining_steps > 0 and not self.stop_requested

    def close(self):
        self._closed = True


def run_generic_loop(loop, context, params, interface="model-context"):
    """Run a chosen callable; reports attest user declarations, not updates."""
    if not callable(loop) or not isinstance(params, dict):
        raise UserError("GENERIC_LOOP_INVALID", "需要用户循环 callable 和参数对象。")
    if not isinstance(interface, str) or interface not in {"context", "model-context"}:
        raise UserError("GENERIC_LOOP_INVALID", "循环签名必须为 context 或 model-context。")
    try:
        try:
            result = loop(context, **params) if interface == "context" else loop(None, context, **params)
        except UserError:
            raise
        except Exception as error:
            raise UserError("GENERIC_LOOP_FAILED", f"用户循环失败：{type(error).__name__}: {error}",
                            "检查用户循环与依赖；用户代码可能已有副作用。",
                            {"completed_steps": context.completed_steps, "total_steps": context.total_steps}) from error
    finally:
        context.close()
    # A stop arriving after the final accepted step does not undo completion.
    # Earlier cooperative stops retain the last reported checkpoint.
    if context.remaining_steps and context.stop_requested:
        raise UserError("USER_INTERRUPTED", "用户循环已响应停止请求。",
                        "最近一次报告的 JSON 状态可作为恢复点；未报告的状态不会保存。",
                        {"completed_steps": context.completed_steps})
    if context._reports == 0:
        raise UserError("GENERIC_LOOP_NO_STEPS", "用户循环没有报告完成的步骤。",
                        data={"completed_steps": context.completed_steps})
    try:
        summary = _json_state(result) if result is not None else None
        result_status = "json" if result is not None else "none"
    except UserError:
        summary = None
        result_status = "not_recorded"
    return {
        "training_mode": "generic-external-loop",
        "loop_interface": interface,
        "completed_steps": context.completed_steps,
        "requested_steps": context.total_steps,
        "termination_reason": "target_reached" if context.remaining_steps == 0 else "loop_returned_before_target",
        "last_metrics": context._last_metrics,
        "result": summary,
        "result_status": result_status,
        "coverage": "进度、指标与状态由用户循环声明；工具不验证计算或参数更新。",
    }
