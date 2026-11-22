"""Capability-driven checks for custom transforms on the executable data path.

This is deliberately independent from extension manifest discovery: custom
transforms are currently wired through ``data.transforms`` as ``module:callable``
steps, while the component catalog does not yet register a transform kind.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError


@dataclass(frozen=True)
class TransformConformanceRuntime:
    add_project_import_path: Callable[[str], None]
    verify_component_references: Callable[..., Any]
    entrypoint: Callable[[str], Callable[..., Any]]
    run_pipeline: Callable[..., dict]
    torch: Any
    tree: Callable[[Any], Any]


def component_test_transform(runtime, request, root, component):
    """Execute a custom transform against a fresh fixture and explicit port claim."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise UserError("PROJECT_ROOT_MISSING", f"项目目录不存在：{root}")
    runtime.add_project_import_path(str(root))

    if not isinstance(component, dict) or component.get("kind") != "transform":
        raise UserError("COMPONENT_TEST_UNSUPPORTED", "符合性声明必须是 transform 类型。")
    runtime.verify_component_references(
        {
            "action": "transform",
            "component": request.get("component"),
            "component_ref": request.get("component_ref"),
        },
        root,
    )
    capabilities = component.get("capabilities", {})
    declaration = capabilities.get("transform") if isinstance(capabilities, dict) else None
    if isinstance(declaration, str):
        state, reason = declaration, None
    elif isinstance(declaration, dict):
        state, reason = declaration.get("state", "unknown"), declaration.get("reason")
    else:
        state, reason = "unknown", "组件未声明 transform 能力。"

    check = {"declared_state": state, "status": "not_tested"}
    if state == "unsupported":
        check.update({"status": "skipped", "reason": reason or "组件声明不支持 transform。"})
        return {
            "status": "passed",
            "coverage": "complete_for_declared_checks",
            "claim_verified": False,
            "capability_results": {"transform": check},
        }
    if state not in {"supported", "conditional"}:
        check["reason"] = reason or f"未知或未支持的能力状态：{state}。"
        return {
            "status": "passed",
            "coverage": "partial",
            "claim_verified": False,
            "capability_results": {"transform": check},
        }

    fixture_ref = component.get("conformance_fixture")
    entrypoint = request.get("component")
    if not isinstance(fixture_ref, str) or ":" not in fixture_ref:
        raise UserError("COMPONENT_FIXTURE_MISSING", "transform 声明需要 module:callable conformance_fixture。")
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise UserError("COMPONENT_ENTRYPOINT_INVALID", "transform component 需要 module:callable 入口。")
    try:
        fixture = runtime.entrypoint(fixture_ref)()
    except Exception as error:
        raise UserError("COMPONENT_FIXTURE_FAILED", f"Transform fixture 执行失败：{type(error).__name__}: {error}") from error
    if not isinstance(fixture, dict) or set(fixture) != {
        "sample",
        "expected_port",
        "expected_output",
    }:
        raise UserError(
            "COMPONENT_FIXTURE_INVALID",
            "Transform fixture 必须返回 sample、expected_port 和 expected_output。",
        )
    expected_port = fixture["expected_port"]
    if not isinstance(expected_port, dict) or not expected_port:
        raise UserError("COMPONENT_FIXTURE_INVALID", "expected_port 必须声明至少一个输出端口断言。")

    specification = [{
        "type": "custom",
        "entrypoint": entrypoint,
        "parameters": request.get("params", {}),
        "output_contract": expected_port,
    }]
    result = runtime.run_pipeline(
        specification,
        fixture["sample"],
        resolve_entrypoint=runtime.entrypoint,
        seed=request.get("seed", 0),
    )
    output = result["output"]
    if hasattr(output, "is_floating_point") and output.is_floating_point():
        if not bool(runtime.torch.isfinite(output.detach()).all().item()):
            raise UserError("COMPONENT_OUTPUT_NONFINITE", "Transform fixture 输出含 NaN 或无穷大。")
    expected_output = fixture["expected_output"]
    if isinstance(output, runtime.torch.Tensor):
        try:
            expected_tensor = runtime.torch.as_tensor(
                expected_output, dtype=output.dtype, device=output.device
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise UserError(
                "COMPONENT_FIXTURE_INVALID",
                f"expected_output 无法转换为预期 Tensor：{error}",
            ) from error
        matches = (
            runtime.torch.allclose(output, expected_tensor, rtol=1e-5, atol=1e-6)
            if output.is_floating_point() or output.is_complex()
            else runtime.torch.equal(output, expected_tensor)
        )
    else:
        matches = output == expected_output
        if not isinstance(matches, bool):
            matches = bool(matches.all()) if hasattr(matches, "all") else False
    if not matches:
        raise UserError(
            "COMPONENT_FIXTURE_MISMATCH",
            "Transform 输出与 fixture 的 expected_output 不一致。",
        )
    runtime.verify_component_references(
        {
            "action": "transform",
            "component": request.get("component"),
            "component_ref": request.get("component_ref"),
        },
        root,
    )
    check.update({
        "status": "passed",
        "output_port": result["output_port"],
        "expected_port": expected_port,
        "step_count": result["step_count"],
        "expected_output": "passed",
    })
    has_fixture_assertion = bool(expected_port)
    return {
        "status": "passed",
        "coverage": "complete_for_fixture_expectations" if has_fixture_assertion else "partial",
        "claim_verified": state == "supported" and has_fixture_assertion,
        "activation_reason": None if state == "supported" else reason or "conditional capability is not activation eligible.",
        "capability_results": {"transform": check},
        "result": {
            "output": runtime.tree(runtime.torch, output),
            "output_port": result["output_port"],
        },
    }
