"""Validate pinned extension references before loading or editing user models."""

import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError


@dataclass(frozen=True)
class ExtensionReferenceRuntime:
    verify_component_distribution: Callable[[dict], Any]
    digest: Callable[[Any], str]
    conformance_suite_version: int
    python_version: str
    torch_version: str


def component_operations(edit):
    if not isinstance(edit, dict):
        return
    if edit.get("action") == "transaction":
        for operation in edit.get("operations", []):
            yield from component_operations(operation)
        return
    reference = edit.get("component_ref")
    if reference is not None:
        yield edit, reference


def module_source_fingerprint(module_name):
    module = importlib.import_module(module_name)
    source_path = inspect.getsourcefile(module) or getattr(module, "__file__", None)
    if not source_path:
        raise UserError(
            "COMPONENT_SOURCE_UNPINNABLE",
            f"无法定位 Python 模块 {module_name} 的源码文件，不能启用注册 Provider。",
            "让 Provider 和 conformance_fixture 位于可读取的 Python 源码文件中。",
        )
    path = Path(source_path)
    if path.suffix == ".pyc":
        source_path = source_path[:-1]
        path = Path(source_path)
    try:
        if not path.is_file() or path.is_symlink():
            raise OSError("源码路径不是普通文件")
        content = path.read_bytes()
    except OSError as error:
        raise UserError(
            "COMPONENT_SOURCE_UNPINNABLE",
            f"无法为 Python 模块 {module_name} 读取稳定源码指纹：{error}",
            "Provider 与 fixture 必须位于当前解释器可读取的普通 .py 文件中。",
        ) from error
    return {
        "module": module_name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def verify_provider_activation(runtime, operation, manifest, reference):
    capability = manifest.get("capabilities", {}).get("layer_edit")
    state = capability if isinstance(capability, str) else capability.get("state") if isinstance(capability, dict) else None
    if state != "supported":
        raise UserError(
            "COMPONENT_PROVIDER_CONDITION_UNVERIFIED"
            if state == "conditional"
            else "COMPONENT_PROVIDER_UNSUPPORTED",
            f"Provider 的 layer_edit 状态为 {state}，当前没有实际编辑上下文验证。",
            "检查组件能力声明；仅 supported 状态可凭 fixture 记录进入执行流程。",
        )
    if not isinstance(manifest.get("conformance_fixture"), str):
        raise UserError(
            "COMPONENT_FIXTURE_MISSING",
            "注册 Provider 没有可用于启用编辑的 conformance_fixture。",
            "增加 fixture 后重新运行 component test。",
        )
    verification = reference.get("verification")
    if not isinstance(verification, dict):
        raise UserError(
            "COMPONENT_PROVIDER_UNVERIFIED",
            f"注册 Provider {manifest.get('component_id')} 尚无有效的 Layer 编辑验证记录。",
            "运行 m-model component test，并为本次相同 payload 提供通过的结构断言。",
        )
    if (
        verification.get("schema_version") != 1
        or verification.get("component_id") != manifest.get("component_id")
        or verification.get("component_version") != manifest.get("component_version")
        or verification.get("manifest_sha256") != reference.get("sha256")
        or not isinstance(verification.get("verifications"), list)
    ):
        raise UserError(
            "COMPONENT_PROVIDER_VERIFICATION_STALE",
            "Layer Edit Provider 的验证记录与清单版本不一致。",
            "重新运行 m-model component test。",
        )

    expected_modules = {
        "provider": manifest.get("entrypoint", "").split(":", 1)[0],
        "fixture": manifest.get("conformance_fixture", "").split(":", 1)[0],
    }
    current_fingerprints = {
        role: module_source_fingerprint(module_name)
        for role, module_name in expected_modules.items()
    }
    payload_sha256 = runtime.digest(operation.get("payload", {}))
    expected = {
        "capability": "layer_edit",
        "status": "verified",
        "coverage": "complete_for_fixture_expectations",
        "suite_version": runtime.conformance_suite_version,
        "payload_sha256": payload_sha256,
        "python_version": runtime.python_version,
        "torch_version": runtime.torch_version,
        "source_fingerprints": current_fingerprints,
    }
    latest_mismatch = None
    for record in reversed(verification["verifications"]):
        if not isinstance(record, dict):
            continue
        mismatch_fields = [
            key for key, expected_value in expected.items()
            if record.get(key) != expected_value
        ]
        if not mismatch_fields:
            return
        if latest_mismatch is None:
            latest_mismatch = {
                "mismatch_fields": mismatch_fields,
                "recorded": {key: record.get(key) for key in expected},
            }
    if latest_mismatch is None:
        latest_mismatch = {
            "mismatch_fields": ["verification_record"],
            "recorded": None,
        }
    raise UserError(
        "COMPONENT_PROVIDER_UNVERIFIED",
        f"Provider {manifest.get('component_id')} 没有与当前 payload、源码和运行环境匹配的通过记录。",
        "运行 m-model component test <id> --param NAME=JSON；比较报告中的差异后，为当前 payload 和运行环境重新取得验证记录。",
        {
            "provider_verification": {
                "state": "stale"
                if latest_mismatch["recorded"] is not None
                else "missing",
                "expected": expected,
                **latest_mismatch,
            }
        },
    )


def verify_component_references(runtime, edit, project_root, allow_unverified=False):
    root = Path(project_root).resolve()

    def read_locked_manifest(reference):
        if not isinstance(reference, dict):
            raise UserError("COMPONENT_REFERENCE_INVALID", "组件引用描述必须是对象。")
        relative = reference.get("path")
        if not isinstance(relative, str):
            raise UserError("COMPONENT_REFERENCE_INVALID", "组件引用缺少相对清单路径。")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.parts[:2] != (".model-maven", "components"):
            raise UserError("COMPONENT_REFERENCE_INVALID", "组件清单路径必须位于项目 .model-maven/components 内。")
        candidate = root / path
        current = root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise UserError("COMPONENT_REFERENCE_INVALID", "组件清单路径不能经过符号链接。")
        if not candidate.is_file():
            raise UserError("COMPONENT_MISSING", f"组件清单不存在：{relative}")
        if candidate.stat().st_size > 256 * 1024:
            raise UserError("COMPONENT_MANIFEST_INVALID", "组件清单超过 256 KiB。")
        contents = candidate.read_bytes()
        actual = hashlib.sha256(contents).hexdigest()
        if actual != reference.get("sha256"):
            raise UserError("COMPONENT_VERSION_CHANGED", f"组件清单已变化：{reference.get('id', relative)}")
        try:
            manifest = json.loads(contents)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UserError("COMPONENT_MANIFEST_INVALID", f"组件清单 JSON 无法读取：{error}") from error
        if (
            not isinstance(manifest, dict)
            or manifest.get("component_id") != reference.get("id")
            or manifest.get("component_version") != reference.get("version")
            or (reference.get("kind") is not None and manifest.get("kind") != reference.get("kind"))
            or (reference.get("entrypoint") is not None and manifest.get("entrypoint") != reference.get("entrypoint"))
            or manifest.get("extension_api_version") != 1
        ):
            raise UserError("COMPONENT_REFERENCE_INVALID", "组件清单身份与已保存的依赖锁不匹配。")
        runtime.verify_component_distribution(manifest)
        return manifest

    for operation, reference in component_operations(edit):
        manifest = read_locked_manifest(reference)
        action = operation.get("action")
        is_provider = action == "custom"
        expected_entrypoint = operation.get("provider") if is_provider else operation.get("component")
        expected_kind = (
            "layer-edit-provider"
            if is_provider
            else action
            if action in {"transform", "dataset", "loss", "metric", "trainer"}
            else "layer"
        )
        if (
            manifest.get("entrypoint") != expected_entrypoint
            or manifest.get("kind") != expected_kind
        ):
            raise UserError("COMPONENT_REFERENCE_INVALID", "组件入口、版本或能力类型与已保存的 Layer 编辑不匹配。")
        dependencies = reference.get("dependencies", [])
        if not isinstance(dependencies, list) or len(dependencies) > 256:
            raise UserError("COMPONENT_REFERENCE_INVALID", "组件依赖锁必须是最多 256 项的数组。")
        seen_dependencies = set()
        for dependency in dependencies:
            read_locked_manifest(dependency)
            key = (dependency.get("id"), dependency.get("version"))
            if key in seen_dependencies:
                raise UserError("COMPONENT_REFERENCE_INVALID", "组件依赖锁包含重复版本。")
            seen_dependencies.add(key)
        if is_provider and not allow_unverified:
            verify_provider_activation(runtime, operation, manifest, reference)


def verify_model_injection_references(runtime, injections, project_root):
    for slot, specification in injections.items():
        if not isinstance(specification, dict) or "component_ref" not in specification:
            continue
        verify_component_references(
            runtime,
            {
                "action": "add",
                "component": specification["factory"],
                "component_ref": specification["component_ref"],
            },
            project_root,
        )


