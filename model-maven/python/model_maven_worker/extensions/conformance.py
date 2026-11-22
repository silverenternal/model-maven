"""Runtime conformance checks for Layer Edit Providers."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError
from ..data.conformance import TransformConformanceRuntime, component_test_transform
from .layer_conformance import LayerConformanceRuntime, component_test_layer
from .trainer_conformance import TrainerConformanceRuntime, component_test_trainer
from .value_conformance import component_test_value


@dataclass(frozen=True)
class ProviderConformanceRuntime:
    verify_component_references: Callable[..., Any]
    entrypoint: Callable[[str], Callable[..., Any]]
    torch: Any
    apply_edit: Callable[[Any, dict], dict]
    digest: Callable[[Any], str]
    conformance_suite_version: int
    module_source_fingerprint: Callable[[str], dict]
    python_version: str
    torch_version: str


@dataclass(frozen=True)
class ExtensionConformanceRuntime:
    torch: Any
    add_project_import_path: Callable[[str], None]
    verify_component_references: Callable[..., Any]
    new_layer: Callable[[dict], Any]
    make_batch: Callable[[dict], Any]
    entrypoint: Callable[[str], Callable[..., Any]]
    tree: Callable[[Any], Any]
    apply_edit: Callable[[Any, dict], dict]
    digest: Callable[[Any], str]
    conformance_suite_version: int
    module_source_fingerprint: Callable[[str], dict]
    python_version: str
    torch_version: str
    run_transform_pipeline: Callable[..., dict]
    run_external_loop: Callable[..., dict] | None = None


def component_test(runtime, request):
    root = Path(request["project_root"]).resolve()
    if not root.is_dir():
        raise UserError("PROJECT_ROOT_MISSING", f"项目目录不存在：{root}")
    runtime.add_project_import_path(str(root))
    runtime.torch.manual_seed(request.get("seed", 42))
    manifest = request.get("component_manifest")
    if not isinstance(manifest, dict):
        raise UserError("COMPONENT_TEST_UNSUPPORTED", "组件清单无效。")
    if manifest.get("kind") == "layer-edit-provider":
        provider_runtime = ProviderConformanceRuntime(
            verify_component_references=runtime.verify_component_references,
            entrypoint=runtime.entrypoint,
            torch=runtime.torch,
            apply_edit=runtime.apply_edit,
            digest=runtime.digest,
            conformance_suite_version=runtime.conformance_suite_version,
            module_source_fingerprint=runtime.module_source_fingerprint,
            python_version=runtime.python_version,
            torch_version=runtime.torch_version,
        )
        return component_test_layer_edit_provider(
            provider_runtime, request, root, manifest
        )
    if manifest.get("kind") == "transform":
        transform_runtime = TransformConformanceRuntime(
            add_project_import_path=runtime.add_project_import_path,
            verify_component_references=runtime.verify_component_references,
            entrypoint=runtime.entrypoint,
            run_pipeline=runtime.run_transform_pipeline,
            torch=runtime.torch,
            tree=runtime.tree,
        )
        return component_test_transform(
            transform_runtime, request, root, manifest
        )
    if manifest.get("kind") in {"dataset", "loss", "metric"}:
        return component_test_value(runtime, request, root, manifest)
    if manifest.get("kind") == "trainer":
        return component_test_trainer(
            TrainerConformanceRuntime(
                torch=runtime.torch,
                entrypoint=runtime.entrypoint,
                run_external_loop=runtime.run_external_loop,
                module_source_fingerprint=runtime.module_source_fingerprint,
                verify_component_references=runtime.verify_component_references,
            ),
            request,
            root,
            manifest,
        )
    if manifest.get("kind") != "layer":
        raise UserError(
            "COMPONENT_TEST_UNSUPPORTED",
            f"当前运行时没有 kind={manifest.get('kind')} 的符合性套件。",
            "可用 component check 检查静态清单。",
        )
    layer_runtime = LayerConformanceRuntime(
        verify_component_references=runtime.verify_component_references,
        new_layer=runtime.new_layer,
        torch=runtime.torch,
        make_batch=runtime.make_batch,
        entrypoint=runtime.entrypoint,
        tree=runtime.tree,
        module_source_fingerprint=runtime.module_source_fingerprint,
    )
    return component_test_layer(layer_runtime, request, root, manifest)


def component_test_layer_edit_provider(runtime, request, root, manifest):
    runtime.verify_component_references(
        {
            "action": "custom",
            "provider": request.get("component"),
            "payload": request.get("params", {}),
            "component_ref": request.get("component_ref"),
        },
        root,
        allow_unverified=True,
    )
    capabilities = manifest.get("capabilities", {})
    declaration = capabilities.get("layer_edit") if isinstance(capabilities, dict) else None
    if isinstance(declaration, str):
        state, reason = declaration, None
    elif isinstance(declaration, dict):
        state, reason = declaration.get("state", "unknown"), declaration.get("reason")
    else:
        state, reason = "unknown", "Provider 未声明 layer_edit 能力。"

    edit_check = {"declared_state": state, "status": "not_tested"}
    source_fingerprints_before = None
    if state == "unsupported":
        edit_check.update({"status": "skipped", "reason": reason or "Provider 声明不支持 Layer 编辑。"})
    elif state in {"supported", "conditional"}:
        fixture_ref = manifest.get("conformance_fixture")
        if not isinstance(fixture_ref, str):
            raise UserError(
                "COMPONENT_FIXTURE_MISSING",
                "Provider 声明支持 Layer 编辑，但没有 conformance_fixture。",
                "提供返回一次性模型和预期结构的 module:callable fixture，再运行 component test。",
            )
        if state == "supported":
            provider_entrypoint = request.get("component")
            if not isinstance(provider_entrypoint, str) or provider_entrypoint.count(":") != 1:
                raise UserError("COMPONENT_ENTRYPOINT_INVALID", "Provider 入口格式无效，无法固定测试源码。")
            provider_module = provider_entrypoint.split(":", 1)[0]
            fixture_module = fixture_ref.split(":", 1)[0]
            source_fingerprints_before = {
                "provider": runtime.module_source_fingerprint(provider_module),
                "fixture": runtime.module_source_fingerprint(fixture_module),
            }
        try:
            fixture = runtime.entrypoint(fixture_ref)()
        except UserError:
            raise
        except Exception as error:
            raise UserError(
                "COMPONENT_FIXTURE_FAILED",
                f"Layer Edit Provider fixture 执行失败：{type(error).__name__}: {error}",
                "fixture 应无外部副作用，并为每次调用创建独立的 nn.Module。",
            ) from error
        if not isinstance(fixture, dict) or not isinstance(fixture.get("model"), runtime.torch.nn.Module):
            raise UserError(
                "COMPONENT_FIXTURE_INVALID",
                "Provider fixture 必须返回包含 torch.nn.Module 的对象：{'model': model, ...}。",
            )
        allowed_fixture_fields = {"model", "expected_result", "expected_modules"}
        unknown_fields = set(fixture) - allowed_fixture_fields
        if unknown_fields:
            raise UserError(
                "COMPONENT_FIXTURE_INVALID",
                f"Provider fixture 不支持字段：{', '.join(sorted(unknown_fields))}。",
            )
        expected_result = fixture.get("expected_result")
        if expected_result is not None and not isinstance(expected_result, dict):
            raise UserError("COMPONENT_FIXTURE_INVALID", "expected_result 必须是 JSON 对象。")
        if expected_result is not None:
            try:
                encoded_expectation = json.dumps(
                    expected_result, ensure_ascii=False, allow_nan=False
                )
            except (TypeError, ValueError) as error:
                raise UserError(
                    "COMPONENT_FIXTURE_INVALID",
                    f"expected_result 必须是有限 JSON 数据：{error}",
                ) from error
            if len(encoded_expectation.encode("utf-8")) > 64 * 1024:
                raise UserError("COMPONENT_FIXTURE_INVALID", "expected_result 超过 64 KiB。")
        expected_modules = fixture.get("expected_modules", {})
        if not isinstance(expected_modules, dict) or any(
            not isinstance(path, str)
            or not path
            or (expected is not None and not isinstance(expected, str))
            for path, expected in expected_modules.items()
        ):
            raise UserError(
                "COMPONENT_FIXTURE_INVALID",
                "expected_modules 必须映射非空模块路径到类名，或映射到 null 表示路径不存在。",
            )

        operation = {
            "action": "custom",
            "provider": request.get("component"),
            "payload": request.get("params", {}),
            "component_ref": request.get("component_ref"),
        }
        effect = runtime.apply_edit(fixture["model"], operation)
        if not isinstance(fixture["model"], runtime.torch.nn.Module):
            raise UserError("COMPONENT_EDIT_MODEL_INVALID", "Provider 编辑后模型不再是 nn.Module。")
        actual_result = effect["result"]
        if expected_result is not None and actual_result != expected_result:
            raise UserError(
                "COMPONENT_FIXTURE_MISMATCH",
                "Provider 返回摘要与 fixture 的 expected_result 不一致。",
                f"预期 {json.dumps(expected_result, ensure_ascii=False)}；收到 {json.dumps(actual_result, ensure_ascii=False)}。",
            )
        module_map = dict(fixture["model"].named_modules())
        module_checks = {}
        for path, expected_type in expected_modules.items():
            actual = module_map.get(path)
            if expected_type is None:
                passed = actual is None
            else:
                qualified_type = f"{type(actual).__module__}.{type(actual).__qualname__}" if actual is not None else None
                passed = actual is not None and expected_type in {
                    type(actual).__name__, qualified_type
                }
            module_checks[path] = {
                "expected": expected_type,
                "actual": (
                    f"{type(actual).__module__}.{type(actual).__qualname__}"
                    if actual is not None
                    else None
                ),
                "status": "passed" if passed else "failed",
            }
            if not passed:
                raise UserError(
                    "COMPONENT_FIXTURE_MISMATCH",
                    f"Provider 编辑后的模块结构不符合 fixture 预期：{path}。",
                    f"预期类型 {expected_type!r}，实际类型 {module_checks[path]['actual']!r}。",
                )

        finite_values = 0
        for name, value in [*fixture["model"].named_parameters(), *fixture["model"].named_buffers()]:
            if (value.is_floating_point() or value.is_complex()) and not bool(
                runtime.torch.isfinite(value.detach()).all().item()
            ):
                raise UserError(
                    "COMPONENT_STATE_NONFINITE",
                    f"Provider 编辑后模型状态 {name} 含 NaN 或无穷大。",
                )
            finite_values += value.numel()
        has_expectations = expected_result is not None or bool(expected_modules)
        edit_check.update({
            "status": "passed",
            "reason": (
                "Provider 在 fixture 模型上执行成功，返回 JSON 摘要且满足显式结构预期。"
                if has_expectations
                else "Provider 调用和 JSON 摘要有效；fixture 未声明结果或结构断言。"
            ),
            "result": actual_result,
            "expected_result": "passed" if expected_result is not None else "not_declared",
            "expected_modules": module_checks,
            "finite_state_values": finite_values,
        })
    elif state not in {"unsupported", "unknown", "supported", "conditional"}:
        edit_check.update({"status": "not_tested", "reason": f"未知能力状态：{state}"})
    else:
        edit_check.update({"reason": reason or "Provider 未声明可验证的 Layer 编辑能力。"})

    capability_results = {"layer_edit": edit_check}
    if isinstance(capabilities, dict):
        for name, capability in capabilities.items():
            if name == "layer_edit":
                continue
            if isinstance(capability, str):
                declared_state, capability_reason = capability, None
            elif isinstance(capability, dict):
                declared_state = capability.get("state", "unknown")
                capability_reason = capability.get("reason")
            else:
                declared_state, capability_reason = "unknown", "能力描述无效。"
            capability_results[name] = {
                "declared_state": declared_state,
                "status": "skipped" if declared_state == "unsupported" else "not_tested",
                "reason": capability_reason
                or "Provider 编辑符合性套件不检查此能力。",
            }
    has_expectations = (
        edit_check["status"] == "passed"
        and (edit_check.get("expected_result") == "passed" or bool(edit_check.get("expected_modules")))
    )
    all_declared_checks_finished = all(
        result["status"] in {"passed", "skipped"}
        for result in capability_results.values()
    )
    if edit_check["status"] == "skipped" and all_declared_checks_finished:
        coverage = "complete_for_declared_checks"
    elif edit_check["status"] == "passed" and has_expectations and all_declared_checks_finished:
        coverage = "complete_for_fixture_expectations"
    else:
        coverage = "partial"
    activation_eligible = (
        state == "supported"
        and edit_check["status"] == "passed"
        and has_expectations
        and all_declared_checks_finished
    )
    verification_evidence = None
    activation_reason = None
    if not has_expectations:
        activation_reason = "fixture 至少需要 expected_result 或 expected_modules 断言。"
    elif not all_declared_checks_finished:
        activation_reason = "仍有其它已声明能力未检查；不能启用 Provider 编辑。"
    elif state != "supported":
        activation_reason = (
            "conditional 状态需要与当前模型条件对应的验证；当前 fixture 检查不会启用它。"
            if state == "conditional"
            else "Provider 没有声明可执行的 layer_edit 能力。"
        )
    if activation_eligible:
        try:
            provider_module = request["component"].split(":", 1)[0]
            fixture_module = manifest["conformance_fixture"].split(":", 1)[0]
            source_fingerprints_after = {
                "provider": runtime.module_source_fingerprint(provider_module),
                "fixture": runtime.module_source_fingerprint(fixture_module),
            }
            if source_fingerprints_before != source_fingerprints_after:
                raise UserError(
                    "COMPONENT_SOURCE_CHANGED_DURING_TEST",
                    "Provider 或 fixture 源码在符合性测试期间发生变化，不能记录可执行验证。",
                    "保留文件后重新运行 component test；本次不会启用 Provider 编辑。",
                    {
                        "before": source_fingerprints_before,
                        "after": source_fingerprints_after,
                    },
                )
            verification_evidence = {
                "payload_sha256": runtime.digest(request.get("params", {})),
                "suite_version": runtime.conformance_suite_version,
                "source_fingerprints": source_fingerprints_before,
                "python_version": runtime.python_version,
                "torch_version": str(runtime.torch.__version__),
            }
        except UserError as error:
            activation_eligible = False
            coverage = "partial"
            activation_reason = str(error)
    return {
        "status": "passed",
        "component": {
            "id": manifest.get("component_id"),
            "version": manifest.get("component_version"),
            "entrypoint": manifest.get("entrypoint"),
        },
        "checks": {"layer_edit": edit_check},
        "capability_results": capability_results,
        "coverage": coverage,
        "activation_eligible": activation_eligible,
        "activation_reason": activation_reason,
        "verification_evidence": verification_evidence,
        "limits": [
            "只在 fixture 提供的一次性模型和本次 payload 上执行 Provider；不测试任意输入。",
            "该检查不验证模型变体/源码写盘、重载、撤销或权重迁移持久化。",
            "没有 fixture 断言时，只验证调用成功和摘要格式，不证明编辑语义正确。",
            "Python Provider 和 fixture 在当前解释器权限下运行；普通进程不是 OS 安全沙箱。",
        ],
    }

