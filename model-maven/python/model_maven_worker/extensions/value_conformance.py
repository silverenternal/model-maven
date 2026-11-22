"""Finite fixture checks for registered dataset, loss, and metric factories."""

import math
from pathlib import Path

from ..protocol import UserError


def component_test_value(runtime, request, root, manifest):
    kind = manifest.get("kind")
    if kind not in {"dataset", "loss", "metric"}:
        raise UserError("COMPONENT_TEST_UNSUPPORTED", f"不支持 kind={kind} 的值组件符合性测试。")
    root = Path(root).resolve()
    if not root.is_dir():
        raise UserError("PROJECT_ROOT_MISSING", f"项目目录不存在：{root}")
    runtime.add_project_import_path(str(root))
    reference = {
        "action": kind,
        "component": request.get("component"),
        "component_ref": request.get("component_ref"),
    }
    runtime.verify_component_references(reference, root)

    capabilities = manifest.get("capabilities", {})
    declaration = capabilities.get(kind) if isinstance(capabilities, dict) else None
    if isinstance(declaration, str):
        state, reason = declaration, None
    elif isinstance(declaration, dict):
        state, reason = declaration.get("state", "unknown"), declaration.get("reason")
    else:
        state, reason = "unknown", "组件没有声明与其 kind 对应的可测试能力。"
    check = {"declared_state": state, "status": "not_tested"}

    if state == "unsupported":
        check.update({"status": "skipped", "reason": reason or f"组件声明不支持 {kind}。"})
        return _report(kind, manifest, check, capabilities)
    if state not in {"supported", "conditional"}:
        check["reason"] = reason or f"能力状态为 {state}；没有执行未声明能力。"
        return _report(kind, manifest, check, capabilities)

    fixture_ref = manifest.get("conformance_fixture")
    component_ref = request.get("component")
    if not isinstance(fixture_ref, str) or fixture_ref.count(":") != 1:
        raise UserError("COMPONENT_FIXTURE_MISSING", f"{kind} supported/conditional 声明需要 conformance_fixture。")
    if not isinstance(component_ref, str) or component_ref.count(":") != 1:
        raise UserError("COMPONENT_ENTRYPOINT_INVALID", "组件入口必须使用 module:callable。")
    entry_module = component_ref.split(":", 1)[0]
    fixture_module = fixture_ref.split(":", 1)[0]
    before = {
        "entrypoint": runtime.module_source_fingerprint(entry_module),
        "fixture": runtime.module_source_fingerprint(fixture_module),
    }
    try:
        fixture = runtime.entrypoint(fixture_ref)()
        factory = runtime.entrypoint(component_ref)
        component = factory(**request.get("params", {}))
    except UserError:
        raise
    except Exception as error:
        raise UserError("COMPONENT_FIXTURE_FAILED", f"{kind} 组件或 fixture 执行失败：{type(error).__name__}: {error}") from error

    if kind == "dataset":
        result = _check_dataset(runtime, component, fixture)
    else:
        result = _check_callable(runtime, kind, component, fixture)
    after = {
        "entrypoint": runtime.module_source_fingerprint(entry_module),
        "fixture": runtime.module_source_fingerprint(fixture_module),
    }
    if before != after:
        raise UserError("COMPONENT_SOURCE_CHANGED", "组件或 fixture 源码在符合性测试期间发生变化。", "重新安装或登记组件后重试。")
    runtime.verify_component_references(reference, root)
    check.update({"status": "passed", **result})
    return _report(kind, manifest, check, capabilities)


def _check_dataset(runtime, dataset, fixture):
    if not isinstance(fixture, dict) or set(fixture) != {"expected_length", "sample_index", "expected_sample"}:
        raise UserError("COMPONENT_FIXTURE_INVALID", "dataset fixture 必须提供 expected_length、sample_index 和 expected_sample。")
    expected_length = fixture["expected_length"]
    sample_index = fixture["sample_index"]
    if isinstance(expected_length, bool) or not isinstance(expected_length, int) or not 1 <= expected_length <= 100_000:
        raise UserError("COMPONENT_FIXTURE_INVALID", "expected_length 必须在 1 到 100000 之间。")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or not 0 <= sample_index < expected_length:
        raise UserError("COMPONENT_FIXTURE_INVALID", "sample_index 必须位于 fixture 声明的数据集范围内。")
    try:
        actual_length = len(dataset)
        if actual_length != expected_length:
            raise UserError("COMPONENT_FIXTURE_MISMATCH", f"Dataset 长度不匹配：预期 {expected_length}，实际 {actual_length}。")
        actual = dataset[sample_index]
    except UserError:
        raise
    except Exception as error:
        raise UserError("COMPONENT_DATASET_FAILED", f"Dataset 必须支持有限 len 与索引读取：{type(error).__name__}: {error}") from error
    _assert_matches(runtime, actual, fixture["expected_sample"], "dataset sample")
    return {"length": actual_length, "sample_index": sample_index, "sample": "passed"}


def _check_callable(runtime, kind, component, fixture):
    if not callable(component):
        raise UserError("COMPONENT_FACTORY_INVALID", f"{kind} 工厂必须返回可调用对象。")
    if not isinstance(fixture, dict) or set(fixture) != {"args", "kwargs", "expected_output"}:
        raise UserError("COMPONENT_FIXTURE_INVALID", f"{kind} fixture 必须提供 args、kwargs 和 expected_output。")
    if not isinstance(fixture["args"], (list, tuple)) or not isinstance(fixture["kwargs"], dict):
        raise UserError("COMPONENT_FIXTURE_INVALID", "args 必须是数组，kwargs 必须是对象。")
    args = _clone_inputs(runtime, fixture["args"], requires_grad=(kind == "loss"))
    kwargs = _clone_inputs(runtime, fixture["kwargs"], requires_grad=(kind == "loss"))
    try:
        if kind == "metric":
            with runtime.torch.no_grad():
                output = component(*args, **kwargs)
        else:
            output = component(*args, **kwargs)
    except Exception as error:
        raise UserError("COMPONENT_CALL_FAILED", f"{kind} fixture 调用失败：{type(error).__name__}: {error}") from error
    if kind == "loss":
        if not isinstance(output, runtime.torch.Tensor) or output.ndim != 0:
            raise UserError("COMPONENT_LOSS_SHAPE_INVALID", "Loss 必须返回标量 Tensor。")
        if not bool(runtime.torch.isfinite(output.detach()).all().item()):
            raise UserError("COMPONENT_OUTPUT_NONFINITE", "Loss 输出含 NaN 或无穷大。")
        _assert_matches(runtime, output.detach(), fixture["expected_output"], "loss output")
        try:
            output.backward()
        except Exception as error:
            raise UserError("COMPONENT_LOSS_BACKWARD_FAILED", f"Loss 反向传播失败：{type(error).__name__}: {error}") from error
        gradients = _floating_gradients(runtime, (args, kwargs))
        if gradients == 0:
            raise UserError("COMPONENT_LOSS_GRADIENT_MISSING", "Loss fixture 未产生任何输入梯度。")
        return {"output": "finite_scalar", "gradient_values": gradients}
    _assert_finite(runtime, output, "metric output")
    _assert_matches(runtime, output, fixture["expected_output"], "metric output")
    return {"output": "finite_and_matched"}


def _clone_inputs(runtime, value, requires_grad, depth=0):
    if depth > 32:
        raise UserError("COMPONENT_FIXTURE_INVALID", "fixture 输入嵌套超过 32 层。")
    if isinstance(value, runtime.torch.Tensor):
        cloned = value.detach().clone()
        if requires_grad and cloned.is_floating_point():
            cloned.requires_grad_(True)
        return cloned
    if isinstance(value, dict):
        return {key: _clone_inputs(runtime, child, requires_grad, depth + 1) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_inputs(runtime, child, requires_grad, depth + 1) for child in value)
    if isinstance(value, list):
        return [_clone_inputs(runtime, child, requires_grad, depth + 1) for child in value]
    return value


def _floating_gradients(runtime, value, depth=0):
    if depth > 32:
        return 0
    if isinstance(value, runtime.torch.Tensor):
        grad = value.grad
        if grad is None:
            return 0
        if not bool(runtime.torch.isfinite(grad).all().item()):
            raise UserError("COMPONENT_GRADIENT_NONFINITE", "Loss fixture 产生了 NaN 或无穷大梯度。")
        return grad.numel()
    if isinstance(value, dict):
        return sum(_floating_gradients(runtime, child, depth + 1) for child in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_floating_gradients(runtime, child, depth + 1) for child in value)
    return 0


def _assert_finite(runtime, value, label, depth=0):
    if depth > 32:
        raise UserError("COMPONENT_OUTPUT_TOO_DEEP", f"{label} 嵌套超过 32 层。")
    if isinstance(value, runtime.torch.Tensor):
        if value.numel() > 1_000_000:
            raise UserError("COMPONENT_OUTPUT_TOO_LARGE", f"{label} 超过 1000000 个 Tensor 值。")
        if (value.is_floating_point() or value.is_complex()) and not bool(runtime.torch.isfinite(value.detach()).all().item()):
            raise UserError("COMPONENT_OUTPUT_NONFINITE", f"{label} 含 NaN 或无穷大。")
    elif isinstance(value, float) and not math.isfinite(value):
        raise UserError("COMPONENT_OUTPUT_NONFINITE", f"{label} 含 NaN 或无穷大。")
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite(runtime, child, label, depth + 1)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_finite(runtime, child, label, depth + 1)


def _assert_matches(runtime, actual, expected, label, depth=0):
    if depth > 32:
        raise UserError("COMPONENT_FIXTURE_INVALID", f"{label} 预期值嵌套超过 32 层。")
    if isinstance(actual, runtime.torch.Tensor):
        try:
            expected_tensor = runtime.torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise UserError("COMPONENT_FIXTURE_INVALID", f"{label} 无法转换为 Tensor：{error}") from error
        if actual.shape != expected_tensor.shape:
            raise UserError("COMPONENT_FIXTURE_MISMATCH", f"{label} shape 不匹配。")
        equal = runtime.torch.allclose(actual, expected_tensor, rtol=1e-5, atol=1e-6) if actual.is_floating_point() or actual.is_complex() else runtime.torch.equal(actual, expected_tensor)
    elif isinstance(actual, dict):
        equal = isinstance(expected, dict) and set(actual) == set(expected)
        if equal:
            for key in actual:
                _assert_matches(runtime, actual[key], expected[key], f"{label}.{key}", depth + 1)
            return
    elif isinstance(actual, (tuple, list)):
        equal = isinstance(expected, (tuple, list)) and len(actual) == len(expected)
        if equal:
            for index, (item, expected_item) in enumerate(zip(actual, expected)):
                _assert_matches(runtime, item, expected_item, f"{label}[{index}]", depth + 1)
            return
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        equal = math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
    else:
        equal = actual == expected
    if not equal:
        raise UserError("COMPONENT_FIXTURE_MISMATCH", f"{label} 与 fixture 预期不一致。")


def _report(kind, manifest, check, capabilities):
    results = {kind: check}
    if isinstance(capabilities, dict):
        for name in capabilities:
            results.setdefault(name, {"declared_state": "not_checked", "status": "not_tested", "reason": "本次 fixture 未检查此能力。"})
    coverage = "complete_for_fixture_expectations"
    if any(item["status"] not in {"passed", "skipped"} for item in results.values()):
        coverage = "partial"
    return {
        "status": "passed",
        "component": {"id": manifest.get("component_id"), "version": manifest.get("component_version"), "kind": kind},
        "coverage": coverage,
        "claim_verified": check["status"] == "passed" and check["declared_state"] == "supported",
        "capability_results": results,
        "limits": [
            "只验证一次受限 fixture；不证明任意数据、所有输入或训练流程正确。",
            "组件代码在当前 Python 解释器权限下执行；普通子进程不是 OS 沙箱。",
        ],
    }
