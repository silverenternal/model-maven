"""Finite checks for registered external-loop trainer components."""

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError


@dataclass(frozen=True)
class TrainerConformanceRuntime:
    torch: Any
    entrypoint: Callable[[str], Callable[..., Any]]
    run_external_loop: Callable[..., dict]
    module_source_fingerprint: Callable[[str], dict]
    verify_component_references: Callable[..., Any]


def component_test_trainer(runtime, request, root, manifest):
    root = Path(root).resolve()
    if not root.is_dir():
        raise UserError("PROJECT_ROOT_MISSING", f"项目目录不存在：{root}")
    runtime.verify_component_references(
        {
            "action": "trainer",
            "component": request.get("component"),
            "component_ref": request.get("component_ref"),
        },
        root,
    )
    component_ref = request.get("component")
    fixture_ref = manifest.get("conformance_fixture")
    if not isinstance(component_ref, str) or component_ref.count(":") != 1:
        raise UserError("COMPONENT_ENTRYPOINT_INVALID", "Trainer 入口必须使用 module:callable。")
    capabilities = manifest.get("capabilities", {})
    fixture_needed = any(
        _capability_state(capabilities, name)[0] in {"supported", "conditional"}
        for name in ("events", "stop", "timeout")
    )
    if fixture_needed and (not isinstance(fixture_ref, str) or fixture_ref.count(":") != 1):
        raise UserError(
            "COMPONENT_FIXTURE_MISSING",
            "events/stop/timeout supported 或 conditional 声明需要 conformance_fixture。",
        )
    entry_module = component_ref.split(":", 1)[0]
    fixture_module = fixture_ref.split(":", 1)[0] if isinstance(fixture_ref, str) else None
    before = {"entrypoint": runtime.module_source_fingerprint(entry_module)}
    if fixture_module:
        before["fixture"] = runtime.module_source_fingerprint(fixture_module)

    fixture = None
    if fixture_needed:
        try:
            fixture = runtime.entrypoint(fixture_ref)()
        except UserError:
            raise
        except Exception as error:
            raise UserError(
                "COMPONENT_FIXTURE_FAILED",
                f"Trainer fixture 执行失败：{type(error).__name__}: {error}",
            ) from error
        if not isinstance(fixture, dict) or set(fixture) - {"events", "stop", "timeout"}:
            raise UserError(
                "COMPONENT_FIXTURE_INVALID",
                "Trainer fixture 仅接受 events、stop 和 timeout 场景。",
            )

    results = {}
    for name in capabilities if isinstance(capabilities, dict) else ():
        state, reason = _capability_state(capabilities, name)
        results[name] = {
            "declared_state": state,
            "status": "not_tested",
            "reason": reason or "Trainer 符合性套件没有该能力的检查器。",
        }
    for name in ("events", "stop", "timeout"):
        if not isinstance(capabilities, dict) or name not in capabilities:
            continue
        state, reason = _capability_state(capabilities, name)
        result = {"declared_state": state, "status": "not_tested"}
        if state == "unsupported":
            result.update({"status": "skipped", "reason": reason or f"Trainer 声明不支持 {name}。"})
        elif state not in {"supported", "conditional"}:
            result["reason"] = reason or f"能力状态为 {state}；没有执行未声明能力。"
        elif not isinstance(fixture, dict) or name not in fixture:
            raise UserError("COMPONENT_FIXTURE_INVALID", f"Trainer fixture 缺少 {name} 场景。")
        elif name == "events":
            result.update(_check_events(runtime, request, component_ref, fixture["events"]))
        elif name == "timeout":
            result.update(_check_timeout(runtime, request, component_ref, fixture["timeout"]))
        else:
            result.update(_check_stop(runtime, request, component_ref, fixture["stop"]))
        if state == "conditional" and result.get("status") == "passed":
            result["status"] = "conditional"
            result["observed_status"] = "passed"
            result["reason"] = (
                f"当前 {name} fixture 通过；声明条件未由通用符合性检查器完整验证，能力仍保持 conditional。"
            )
        results[name] = result

    after = {"entrypoint": runtime.module_source_fingerprint(entry_module)}
    if fixture_module:
        after["fixture"] = runtime.module_source_fingerprint(fixture_module)
    if before != after:
        raise UserError("COMPONENT_SOURCE_CHANGED", "Trainer 或 fixture 源码在符合性测试期间发生变化。")
    runtime.verify_component_references(
        {
            "action": "trainer",
            "component": component_ref,
            "component_ref": request.get("component_ref"),
        },
        root,
    )
    coverage = (
        "complete_for_declared_checks"
        if all(item["status"] in {"passed", "skipped"} for item in results.values())
        else "partial"
    )
    return {
        "status": "passed",
        "component": {
            "id": manifest.get("component_id"),
            "version": manifest.get("component_version"),
            "entrypoint": component_ref,
        },
        "capability_results": results,
        "coverage": coverage,
        "limits": [
            "事件检查覆盖 fixture 声明的有限步数和指标；不验证用户循环中的 backward 或 optimizer 正确性。",
            "stop 检查只验证组件轮询 stop_requested 并在步骤边界停止。",
            "timeout fixture 验证协作式 deadline 与步骤边界停止；CLI watchdog 仍是进程级兜底，长时间不返回的单步无法保证 checkpoint。",
            "Python 入口在当前解释器权限下执行；普通进程不是 OS 安全沙箱。",
        ],
    }


def _capability_state(capabilities, name):
    declaration = capabilities.get(name) if isinstance(capabilities, dict) else None
    if isinstance(declaration, str):
        return declaration, None
    if isinstance(declaration, dict):
        return declaration.get("state", "unknown"), declaration.get("reason")
    return "unknown", "能力没有有效声明。"


def _check_events(runtime, request, component_ref, fixture):
    if not isinstance(fixture, dict) or set(fixture) != {
        "steps", "expected_mean_loss", "expected_metrics"
    }:
        raise UserError(
            "COMPONENT_FIXTURE_INVALID",
            "events fixture 必须提供 steps、expected_mean_loss 和 expected_metrics。",
        )
    steps = fixture["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 16:
        raise UserError("COMPONENT_FIXTURE_INVALID", "events.steps 必须在 1 到 16 之间。")
    expected_loss = fixture["expected_mean_loss"]
    expected_metrics = fixture["expected_metrics"]
    if not _finite_number(expected_loss) or not isinstance(expected_metrics, dict):
        raise UserError("COMPONENT_FIXTURE_INVALID", "events 预期 loss/metrics 格式无效。")
    loop = _make_loop(runtime, request, component_ref)
    training_request = _training_request(request, component_ref, steps)
    try:
        result = runtime.run_external_loop(None, training_request, loop)
    except UserError:
        raise
    except Exception as error:
        raise UserError(
            "COMPONENT_TRAINER_FAILED",
            f"Trainer 事件 fixture 失败：{type(error).__name__}: {error}",
        ) from error
    if result.get("completed_steps") != steps:
        raise UserError("COMPONENT_FIXTURE_MISMATCH", "Trainer 报告的步骤数与 fixture 不一致。")
    if not math.isclose(result.get("mean_loss", math.nan), expected_loss, rel_tol=1e-6, abs_tol=1e-7):
        raise UserError("COMPONENT_FIXTURE_MISMATCH", "Trainer 事件的平均 loss 与 fixture 不一致。")
    actual_metrics = result.get("mean_metrics", {})
    if set(actual_metrics) != set(expected_metrics) or any(
        not _finite_number(value)
        or not math.isclose(actual_metrics[name], value, rel_tol=1e-6, abs_tol=1e-7)
        for name, value in expected_metrics.items()
    ):
        raise UserError("COMPONENT_FIXTURE_MISMATCH", "Trainer 事件指标与 fixture 不一致。")
    return {
        "status": "passed",
        "completed_steps": steps,
        "mean_loss": result["mean_loss"],
        "metrics": actual_metrics,
    }


def _check_stop(runtime, request, component_ref, fixture):
    if not isinstance(fixture, dict) or set(fixture) != {"steps"}:
        raise UserError("COMPONENT_FIXTURE_INVALID", "stop fixture 必须提供 steps。")
    steps = fixture["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 16:
        raise UserError("COMPONENT_FIXTURE_INVALID", "stop.steps 必须在 1 到 16 之间。")
    loop = _make_loop(runtime, request, component_ref)
    with tempfile.TemporaryDirectory(prefix="model-maven-trainer-stop-") as directory:
        stop_file = Path(directory) / "stop.request"
        stop_file.write_text("stop", encoding="utf-8")
        training_request = _training_request(request, component_ref, steps)
        training_request["stop_file"] = str(stop_file)
        try:
            result = runtime.run_external_loop(None, training_request, loop)
        except UserError as error:
            if error.code != "USER_INTERRUPTED":
                raise
            return {"status": "passed", "stop": "observed_before_first_update"}
    if result.get("completed_steps", 0) != 0:
        raise UserError(
            "COMPONENT_FIXTURE_MISMATCH",
            "stop 文件存在时 Trainer 仍报告了更新；请在边界检查 context.stop_requested。",
        )
    raise UserError("COMPONENT_FIXTURE_MISMATCH", "Trainer 未响应已存在的 stop 请求。")


def _check_timeout(runtime, request, component_ref, fixture):
    if not isinstance(fixture, dict) or set(fixture) != {"steps", "timeout_seconds"}:
        raise UserError(
            "COMPONENT_FIXTURE_INVALID",
            "timeout fixture 必须提供 steps 和 timeout_seconds。",
        )
    steps = fixture["steps"]
    timeout = fixture["timeout_seconds"]
    if type(steps) is not int or not 2 <= steps <= 16:
        raise UserError("COMPONENT_FIXTURE_INVALID", "timeout.steps 必须在 2 到 16 之间。")
    if type(timeout) is not int or not 1 <= timeout <= 5:
        raise UserError("COMPONENT_FIXTURE_INVALID", "timeout.timeout_seconds 必须在 1 到 5 秒之间。")
    training_request = _training_request(request, component_ref, steps)
    training_request["runtime"] = {"timeout_seconds": timeout}
    started = time.monotonic()
    try:
        result = runtime.run_external_loop(
            None,
            training_request,
            _make_loop(runtime, request, component_ref),
        )
    except UserError:
        raise
    except Exception as error:
        raise UserError(
            "COMPONENT_TRAINER_FAILED",
            f"Trainer timeout fixture 失败：{type(error).__name__}: {error}",
        ) from error
    elapsed = time.monotonic() - started
    completed = result.get("completed_steps")
    if (
        result.get("termination_reason") != "timeout"
        or type(completed) is not int
        or not 1 <= completed < steps
        or not math.isfinite(elapsed)
        or elapsed > timeout + 2.0
    ):
        raise UserError(
            "COMPONENT_FIXTURE_MISMATCH",
            "Trainer 未在 cooperative deadline 到期后，于有限步骤边界停止。",
            "检查 context.timeout_requested、remaining_time_seconds 和 report_step() 的返回值。",
            {"completed_steps": completed, "requested_steps": steps, "elapsed_seconds": elapsed},
        )
    return {
        "status": "passed",
        "completed_steps": completed,
        "elapsed_seconds": elapsed,
        "deadline_seconds": timeout,
        "termination_reason": "timeout",
    }


def _make_loop(runtime, request, component_ref):
    try:
        loop = runtime.entrypoint(component_ref)(**request.get("params", {}))
    except UserError:
        raise
    except Exception as error:
        raise UserError(
            "COMPONENT_FACTORY_FAILED",
            f"Trainer 工厂执行失败：{type(error).__name__}: {error}",
        ) from error
    if not callable(loop):
        raise UserError("COMPONENT_FACTORY_INVALID", "Trainer 工厂必须返回 external-loop callable。")
    return loop


def _training_request(request, component_ref, steps):
    return {
        "operation": "training_probe",
        "training_mode": "external-loop",
        "training_loop": component_ref,
        "training_params": {},
        "steps": steps,
        "seed": request.get("seed", 42),
    }


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
