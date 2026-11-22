"""Runtime conformance checks for registered Layer components."""

import math
from dataclasses import dataclass
from typing import Any, Callable

from ..protocol import UserError


@dataclass(frozen=True)
class LayerConformanceRuntime:
    verify_component_references: Callable[..., Any]
    new_layer: Callable[[dict], Any]
    torch: Any
    make_batch: Callable[..., Any]
    entrypoint: Callable[[str], Callable[..., Any]]
    tree: Callable[..., Any]
    module_source_fingerprint: Callable[[str], Any]


def component_test_layer(runtime, request, root, manifest):
    component_ref = request.get("component")
    if not isinstance(component_ref, str) or component_ref.count(":") != 1:
        raise UserError("COMPONENT_ENTRYPOINT_INVALID", "组件入口必须使用 module:callable。")
    component_module = component_ref.split(":", 1)[0]
    construct_state, construct_reason = _capability_declaration(
        manifest.get("capabilities", {}), "construct", "implicit_required"
    )
    if construct_state == "unsupported":
        raise UserError(
            "COMPONENT_CAPABILITY_CONTRADICTION",
            "Layer 工厂是 component test 的必要入口，但该组件声明不支持 construct。",
            construct_reason or "将 construct 改为 supported/conditional，或不要注册为 Layer。",
        )
    source_modules = {"entrypoint": component_module}
    batch_ref = request.get("batch")
    if isinstance(batch_ref, str) and batch_ref.count(":") == 1:
        batch_module = batch_ref.split(":", 1)[0]
        if batch_module != component_module:
            source_modules["batch"] = batch_module
    runtime.verify_component_references(
        {
            "action": "add",
            "component": request.get("component"),
            "component_ref": request.get("component_ref"),
        },
        root,
    )
    source_before = {}
    fingerprint_by_module = {}
    for role, module_name in source_modules.items():
        if module_name not in fingerprint_by_module:
            fingerprint_by_module[module_name] = runtime.module_source_fingerprint(module_name)
        source_before[role] = fingerprint_by_module[module_name]
    layer = runtime.new_layer({
        "component": component_ref,
        "params": request.get("params", {}),
    })
    layer.eval()
    parameter_count = sum(parameter.numel() for parameter in layer.parameters())
    buffer_count = sum(buffer.numel() for buffer in layer.buffers())
    checked_values = 0
    for name, value in [*layer.named_parameters(), *layer.named_buffers()]:
        if (value.is_floating_point() or value.is_complex()) and not bool(
            runtime.torch.isfinite(value.detach()).all().item()
        ):
            raise UserError(
                "COMPONENT_STATE_NONFINITE",
                f"组件构造后状态 {name} 含 NaN 或无穷大。",
                "检查工厂初始化及参数 Schema；本次测试没有修改工程文件。",
            )
        checked_values += value.numel()

    capabilities = manifest.get("capabilities", {})
    capability_results = {}
    if isinstance(capabilities, dict):
        for name, declaration in capabilities.items():
            if isinstance(declaration, str):
                state, reason = declaration, None
            elif isinstance(declaration, dict):
                state, reason = declaration.get("state", "unknown"), declaration.get("reason")
            else:
                state, reason = "unknown", "能力描述无效。"
            capability_results[name] = {
                "declared_state": state,
                "status": "not_tested",
                "reason": reason or "当前 Layer 符合性套件没有该能力的通用检查器。",
            }
    state_roundtrip = _check_state_roundtrip(
        runtime, layer, request, root, manifest, capabilities
    )
    if state_roundtrip is not None:
        capability_results["state_roundtrip"] = state_roundtrip
    construct_result = {
        "declared_state": construct_state,
        "status": (
            "passed"
            if construct_state in {"supported", "implicit_required"}
            else "conditional"
            if construct_state == "conditional"
            else "not_tested"
        ),
        "reason": construct_reason or "本次工厂调用构造出 torch.nn.Module。",
    }
    if construct_state == "implicit_required":
        construct_result.update({
            "status": "passed",
            "reason": "Layer 工厂构造出 torch.nn.Module。",
        })
    elif construct_state in {"conditional", "unknown"}:
        construct_result["observed_status"] = "passed"
    capability_results["construct"] = construct_result
    forward_capability = capabilities.get("forward_probe") if isinstance(capabilities, dict) else None
    if isinstance(forward_capability, str):
        forward_state = forward_capability
        forward_reason = None
    elif isinstance(forward_capability, dict):
        forward_state = forward_capability.get("state", "unknown")
        forward_reason = forward_capability.get("reason")
    else:
        forward_state = "unknown"
        forward_reason = "组件未声明 forward_probe 能力。"

    forward = {
        "state": forward_state,
        "status": "conditional" if forward_state == "conditional" else "not_run",
    }
    if forward_state == "conditional":
        forward["reason"] = forward_reason or "声明的条件未由通用符合性检查器验证。"
    if forward_state == "unsupported":
        forward.update({"status": "skipped", "reason": forward_reason or "组件声明不支持前向试跑。"})
    elif forward_state in {"supported", "conditional"}:
        if not request.get("batch"):
            forward["reason"] = "提供 --batch module:factory 才能验证真实 Layer 输入。"
        else:
            args, kwargs, _ = runtime.make_batch(request, runtime.torch, runtime.entrypoint)
            try:
                with runtime.torch.no_grad():
                    output = layer(*args, **kwargs)
                budget = [10000]

                def check_finite(value, depth=0):
                    if depth > 32:
                        raise UserError("COMPONENT_OUTPUT_TOO_DEEP", "组件输出嵌套超过 32 层。")
                    budget[0] -= 1
                    if budget[0] < 0:
                        raise UserError("COMPONENT_OUTPUT_TOO_LARGE", "组件输出节点超过 10,000 个。")
                    if isinstance(value, runtime.torch.Tensor):
                        if (value.is_floating_point() or value.is_complex()) and not bool(
                            runtime.torch.isfinite(value.detach()).all().item()
                        ):
                            raise UserError(
                                "COMPONENT_OUTPUT_NONFINITE",
                                "组件前向输出包含 NaN 或无穷大。",
                            )
                    elif isinstance(value, float) and not math.isfinite(value):
                        raise UserError(
                            "COMPONENT_OUTPUT_NONFINITE",
                            "组件前向输出包含 NaN 或无穷大。",
                        )
                    elif isinstance(value, dict):
                        for child in value.values():
                            check_finite(child, depth + 1)
                    elif isinstance(value, (tuple, list)):
                        for child in value:
                            check_finite(child, depth + 1)

                check_finite(output)
                forward.update({
                    "status": "passed" if forward_state == "supported" else "conditional",
                    "inputs": runtime.tree(runtime.torch, {"args": args, "kwargs": kwargs}),
                    "outputs": runtime.tree(runtime.torch, output),
                    "mode": "eval",
                })
                if forward_state == "conditional":
                    forward["observed_status"] = "passed"
                    forward["reason"] = (
                        "当前批次前向通过；声明条件未由通用符合性检查器完整验证，能力仍保持 conditional。"
                    )
            except UserError:
                raise
            except Exception as error:
                raise UserError(
                    "COMPONENT_FORWARD_FAILED",
                    f"组件前向失败：{type(error).__name__}: {error}",
                    "当前批次没有覆盖所有输入和控制流；请用代表性输入重复测试。",
                ) from error
    else:
        forward["reason"] = forward_reason or "能力状态为 unknown；未对未声明能力执行测试。"
    if isinstance(capabilities, dict) and "forward_probe" in capabilities:
        capability_results["forward_probe"] = {
            "declared_state": forward_state,
            "status": forward["status"],
            **(
                {"observed_status": forward["observed_status"]}
                if forward.get("observed_status")
                else {}
            ),
            **({"reason": forward["reason"]} if forward.get("reason") else {}),
        }

    source_after = {}
    fingerprint_by_module = {}
    for role, module_name in source_modules.items():
        if module_name not in fingerprint_by_module:
            fingerprint_by_module[module_name] = runtime.module_source_fingerprint(module_name)
        source_after[role] = fingerprint_by_module[module_name]
    if source_before != source_after:
        raise UserError(
            "COMPONENT_SOURCE_CHANGED_DURING_TEST",
            "组件入口源码在符合性测试期间发生变化，不能记录本次检查结果。",
            "保留文件后重新运行 component test。",
            {"before": source_before, "after": source_after},
        )
    runtime.verify_component_references(
        {
            "action": "add",
            "component": component_ref,
            "component_ref": request.get("component_ref"),
        },
        root,
    )

    coverage = "complete_for_declared_checks"
    if any(
        result["status"] not in {"passed", "skipped"}
        for result in capability_results.values()
    ):
        coverage = "partial"
    return {
        "status": "passed",
        "component": {
            "id": manifest.get("component_id"),
            "version": manifest.get("component_version"),
            "entrypoint": manifest.get("entrypoint"),
        },
        "checks": {
            "factory_import": "passed",
            "constructs_nn_module": "passed",
            "constructed_state_finite": "passed",
            "forward_probe": forward,
            "state_roundtrip": {
                "status": capability_results.get("state_roundtrip", {}).get(
                    "status", "not_tested"
                ),
                **(
                    {"reason": capability_results["state_roundtrip"]["reason"]}
                    if capability_results.get("state_roundtrip", {}).get("reason")
                    else {}
                ),
            },
        },
        "capability_results": capability_results,
        "module": {
            "type": f"{type(layer).__module__}.{type(layer).__qualname__}",
            "parameters": parameter_count,
            "buffers": buffer_count,
            "checked_state_values": checked_values,
        },
        "coverage": coverage,
        "source_fingerprints": {"entrypoint": source_before},
        "limits": [
            "只验证本次选择的参数和批次，不证明任意输入、训练梯度或所有分支正确。",
            "未声明的能力不会自动显示为已验证。",
            "Python 工厂在当前解释器权限下运行；普通进程不是 OS 安全沙箱。",
        ],
    }


def _capability_declaration(capabilities, name, default_state="unknown"):
    if not isinstance(capabilities, dict):
        return default_state, None
    declaration = capabilities.get(name)
    if isinstance(declaration, str):
        return declaration, None
    if isinstance(declaration, dict):
        return declaration.get("state", default_state), declaration.get("reason")
    return default_state, None


def _check_state_roundtrip(runtime, layer, request, root, manifest, capabilities):
    declaration = capabilities.get("state_roundtrip") if isinstance(capabilities, dict) else None
    if isinstance(declaration, str):
        state, reason = declaration, None
    elif isinstance(declaration, dict):
        state, reason = declaration.get("state", "unknown"), declaration.get("reason")
    else:
        return None
    check = {"declared_state": state, "status": "not_tested"}
    if state == "unsupported":
        check.update({"status": "skipped", "reason": reason or "组件声明不支持状态往返。"})
        return check
    if state not in {"supported", "conditional"}:
        check["reason"] = reason or "能力状态为 unknown；未执行未声明的状态往返。"
        return check

    fixture_ref = manifest.get("conformance_fixture")
    if not isinstance(fixture_ref, str) or fixture_ref.count(":") != 1:
        raise UserError(
            "COMPONENT_FIXTURE_MISSING",
            "state_roundtrip supported/conditional 声明需要 conformance_fixture。",
        )
    component_ref = request.get("component")
    if not isinstance(component_ref, str) or component_ref.count(":") != 1:
        raise UserError("COMPONENT_ENTRYPOINT_INVALID", "组件入口必须使用 module:callable。")
    fixture_module = fixture_ref.split(":", 1)[0]
    component_module = component_ref.split(":", 1)[0]
    before = {
        "entrypoint": runtime.module_source_fingerprint(component_module),
        "fixture": runtime.module_source_fingerprint(fixture_module),
    }
    try:
        fixture = runtime.entrypoint(fixture_ref)()
        if not isinstance(fixture, dict) or set(fixture) != {"args", "kwargs"}:
            raise UserError(
                "COMPONENT_FIXTURE_INVALID",
                "state_roundtrip fixture 必须返回 args 和 kwargs。",
            )
        if not isinstance(fixture["args"], (list, tuple)) or not isinstance(fixture["kwargs"], dict):
            raise UserError("COMPONENT_FIXTURE_INVALID", "fixture args 必须是数组，kwargs 必须是对象。")
        args = fixture["args"]
        kwargs = fixture["kwargs"]
        with runtime.torch.no_grad():
            expected = layer(*args, **kwargs)
            state_dict = layer.state_dict()
        if not isinstance(state_dict, dict) or len(state_dict) > 10_000:
            raise UserError("COMPONENT_STATE_INVALID", "state_dict 必须是有界 mapping。")
        budget = [10_000, 1_000_000]
        for key, value in state_dict.items():
            if not isinstance(key, str):
                raise UserError("COMPONENT_STATE_INVALID", "state_dict 的顶层 key 必须是字符串。")
            _check_state_value(runtime, value, budget)
        if budget[1] < 0:
            raise UserError("COMPONENT_STATE_TOO_LARGE", "state_dict 超过 1000000 个 Tensor 值。")
        restored = runtime.new_layer({
            "component": component_ref,
            "params": request.get("params", {}),
        })
        restored.load_state_dict(state_dict, strict=True)
        with runtime.torch.no_grad():
            actual = restored(*args, **kwargs)
        if not _same_output(runtime, expected, actual):
            raise UserError("COMPONENT_STATE_ROUNDTRIP_MISMATCH", "load_state_dict 后输出与原组件不一致。")
    except UserError:
        raise
    except Exception as error:
        raise UserError(
            "COMPONENT_STATE_ROUNDTRIP_FAILED",
            f"组件 state_dict/load_state_dict 往返失败：{type(error).__name__}: {error}",
        ) from error
    after = {
        "entrypoint": runtime.module_source_fingerprint(component_module),
        "fixture": runtime.module_source_fingerprint(fixture_module),
    }
    if before != after:
        raise UserError("COMPONENT_SOURCE_CHANGED", "组件或 fixture 源码在符合性测试期间发生变化。")
    runtime.verify_component_references(
        {"action": "add", "component": component_ref, "component_ref": request.get("component_ref")},
        root,
    )
    check.update({
        "status": "passed" if state == "supported" else "conditional",
        "state_entries": len(state_dict),
        "tensor_values": 1_000_000 - budget[1],
    })
    if state == "conditional":
        check["observed_status"] = "passed"
        check["reason"] = (
            "当前 fixture 的状态往返通过；声明条件未由通用符合性检查器完整验证，能力仍保持 conditional。"
        )
    return check


def _check_state_value(runtime, value, budget, depth=0):
    budget[0] -= 1
    if budget[0] < 0 or depth > 32:
        raise UserError("COMPONENT_STATE_TOO_LARGE", "state_dict 超过 10000 个节点或 32 层嵌套。")
    if isinstance(value, runtime.torch.Tensor):
        budget[1] -= value.numel()
        if budget[1] < 0:
            raise UserError("COMPONENT_STATE_TOO_LARGE", "state_dict 超过 1000000 个 Tensor 值。")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            runtime.torch.isfinite(value.detach()).all().item()
        ):
            raise UserError("COMPONENT_STATE_NONFINITE", "state_dict 包含 NaN 或无穷大。")
        return
    if value is None or type(value) in {bool, int}:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UserError("COMPONENT_STATE_NONFINITE", "state_dict 包含 NaN 或无穷大。")
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 64 * 1024:
            raise UserError("COMPONENT_STATE_TOO_LARGE", "state_dict 字符串超过 64 KiB。")
        return
    if isinstance(value, dict):
        if len(value) > 10_000 or any(not isinstance(key, str) for key in value):
            raise UserError("COMPONENT_STATE_INVALID", "state_dict 内部 mapping 必须使用有界字符串键。")
        for child in value.values():
            _check_state_value(runtime, child, budget, depth + 1)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > 10_000:
            raise UserError("COMPONENT_STATE_TOO_LARGE", "state_dict 序列超过 10000 项。")
        for child in value:
            _check_state_value(runtime, child, budget, depth + 1)
        return
    raise UserError("COMPONENT_STATE_INVALID", "state_dict 包含不支持的状态值。")


def _same_output(runtime, expected, actual, depth=0):
    if depth > 32:
        return False
    if isinstance(expected, runtime.torch.Tensor) or isinstance(actual, runtime.torch.Tensor):
        return (
            isinstance(expected, runtime.torch.Tensor)
            and isinstance(actual, runtime.torch.Tensor)
            and expected.shape == actual.shape
            and expected.dtype == actual.dtype
            and bool(runtime.torch.equal(expected, actual))
        )
    if isinstance(expected, dict):
        return isinstance(actual, dict) and expected.keys() == actual.keys() and all(
            _same_output(runtime, expected[key], actual[key], depth + 1) for key in expected
        )
    if isinstance(expected, (tuple, list)):
        return isinstance(actual, type(expected)) and len(expected) == len(actual) and all(
            _same_output(runtime, left, right, depth + 1) for left, right in zip(expected, actual)
        )
    return type(expected) is type(actual) and expected == actual
