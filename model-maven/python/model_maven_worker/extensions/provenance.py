"""Verify installed extension provenance against its discovered metadata."""

import hashlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from ..environment import distribution_name_key
from ..protocol import UserError
from .catalog import (
    distribution_manifest_path,
    distribution_module_source,
    entrypoint_module,
    safe_distribution_file,
)


@dataclass(frozen=True)
class ComponentProvenanceRuntime:
    search_path_entries: tuple[str, ...]
    current_directory: str
    loaded_module_path: Callable[[str], str | None]
    prepend_import_path: Callable[[str], None]


def verify_component_distribution(runtime, manifest):
    source = manifest.get("package_source")
    if source is None:
        return None
    if not isinstance(source, dict):
        raise UserError("COMPONENT_PACKAGE_SOURCE_INVALID", "package_source 必须是对象。")
    name = source.get("distribution")
    version = source.get("distribution_version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise UserError("COMPONENT_PACKAGE_SOURCE_INVALID", "扩展包来源缺少 distribution 名称或版本。")
    recorded_root = source.get("distribution_root")
    roots = []
    for item in runtime.search_path_entries:
        try:
            root = Path(item or runtime.current_directory).resolve(strict=True)
            if root.is_dir() and str(root) not in roots:
                roots.append(str(root))
        except OSError:
            continue
    if isinstance(recorded_root, str):
        try:
            root = Path(recorded_root).resolve(strict=True)
            if root.is_dir() and str(root) not in roots:
                roots.append(str(root))
        except OSError:
            pass

    expected_name = distribution_name_key(name)
    expected_version = version
    manifest_path = source.get("manifest_file")
    expected_manifest_hash = source.get("manifest_sha256")
    expected_modules = source.get("module_sources")
    candidates = []
    for distribution in importlib.metadata.distributions(path=roots):
        if (
            distribution_name_key(distribution.metadata.get("Name", "")) == expected_name
            and distribution.version == expected_version
        ):
            candidates.append(distribution)
    failures = []
    for distribution in candidates:
        try:
            root = Path(distribution.locate_file("")).resolve(strict=True)
            files = distribution.files
            if files is None:
                raise ValueError("distribution 没有可验证的文件清单")
            declared_files = {PurePosixPath(str(item)).as_posix() for item in files}
            entry_value = manifest.get("entrypoint")
            entry_module = entrypoint_module(entry_value)
            matching_entry = any(
                point.group == "model_maven.components"
                and point.name == source.get("entry_point")
                and point.value.strip() == entry_value
                for point in distribution.entry_points
            )
            if not matching_entry:
                raise ValueError("扩展 entry point 已变化或缺失")
            manifest_file = safe_distribution_file(root, manifest_path, declared_files, 256 * 1024)
            if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != expected_manifest_hash:
                raise ValueError("扩展 manifest 内容已变化")
            if not isinstance(expected_modules, dict) or "entrypoint" not in expected_modules:
                raise ValueError("缺少入口模块源码指纹")
            expected_role_modules = {
                "entrypoint": manifest.get("entrypoint", "").split(":", 1)[0],
            }
            if manifest.get("conformance_fixture"):
                expected_role_modules["fixture"] = manifest["conformance_fixture"].split(":", 1)[0]
            if set(expected_modules) != set(expected_role_modules):
                raise ValueError("模块源码指纹角色与 manifest 不匹配")
            for role, fingerprint in expected_modules.items():
                if role not in {"entrypoint", "fixture"} or not isinstance(fingerprint, dict):
                    raise ValueError("模块源码指纹格式无效")
                module_file = safe_distribution_file(root, fingerprint.get("file"), declared_files, 8 * 1024 * 1024)
                expected_file = distribution_module_source(
                    root, expected_role_modules[role], declared_files
                )
                if fingerprint.get("file") != expected_file["file"]:
                    raise ValueError(f"{role} 源码路径与 manifest 入口不匹配")
                if hashlib.sha256(module_file.read_bytes()).hexdigest() != fingerprint.get("sha256"):
                    raise ValueError(f"{role} 模块源码已变化")
            if distribution_manifest_path(
                expected_modules["entrypoint"]["file"], source.get("entry_point", "")
            ) != manifest_path:
                raise ValueError("静态清单路径与 entry point 不匹配")
            manifest_module = Path(expected_modules["entrypoint"]["file"])
            absolute_module = root.joinpath(*PurePosixPath(manifest_module.as_posix()).parts).resolve(strict=True)
            loaded_file = runtime.loaded_module_path(entry_module)
            if loaded_file and Path(loaded_file).resolve() != absolute_module:
                raise ValueError("入口模块已从另一个位置加载；拒绝使用来源不明的模块")
            runtime.prepend_import_path(str(root))
            return str(root)
        except Exception as error:
            failures.append(str(error))
    reason = failures[0] if failures else "指定版本 distribution 当前不可用"
    raise UserError(
        "COMPONENT_PACKAGE_SOURCE_MISMATCH",
        f"组件 {manifest.get('component_id')} 的扩展包来源无法验证：{reason}。",
        "恢复记录的 Python distribution/源码版本后重试；发现和检查不会自动安装或升级包。",
    )
