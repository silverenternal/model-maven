"""Static discovery of installed and project-local component packages."""

import hashlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

from ..environment import distribution_name_key
from ..protocol import UserError


class ExternalDistributionResource(ValueError):
    """A declared package file resolves outside its owning distribution."""
from .metadata import validate_manifest


def entrypoint_module(value):
    if not isinstance(value, str) or value.count(":") != 1:
        raise ValueError("entry point 必须使用标准 module:callable 格式")
    module, attribute = (part.strip() for part in value.split(":", 1))
    if (
        not module
        or not attribute
        or not all(part.isidentifier() for part in module.split("."))
        or not all(part.isidentifier() for part in attribute.split("."))
    ):
        raise ValueError("entry point 只能包含 Python 模块与属性标识符")
    return module


def distribution_manifest_path(module_source, entry_name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", entry_name):
        raise ValueError("entry point 名称只能包含字母、数字、点、下划线和连字符")
    source = PurePosixPath(module_source)
    if source.name != "__init__.py" and source.suffix != ".py":
        raise ValueError("entry point 源文件不是 Python 源文件")
    return (source.parent / "__model_maven__" / f"{entry_name}.json").as_posix()


def safe_distribution_file(root, relative, declared_files, maximum_bytes):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("文件路径必须是相对 POSIX 路径")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("文件路径不能包含绝对路径或目录跳转")
    normalized = path.as_posix()
    if normalized not in declared_files:
        raise ValueError("文件未列在 distribution 的文件清单中")
    candidate = root.joinpath(*path.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("distribution 声明的文件不存在") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        if candidate.is_symlink():
            raise ExternalDistributionResource(
                f"{normalized} 指向 distribution 外部资源 {resolved}"
            ) from error
        raise ValueError("文件目标离开 distribution 目录") from error
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            try:
                current.resolve(strict=True).relative_to(root)
            except ValueError as error:
                raise ExternalDistributionResource(
                    f"{normalized} 指向 distribution 外部资源 {current.resolve(strict=True)}"
                ) from error
            raise ValueError("拒绝读取 distribution 内的符号链接")
    if not resolved.is_file():
        raise ValueError("目标不是普通文件")
    size = resolved.stat().st_size
    if size > maximum_bytes:
        raise ValueError(f"文件超过 {maximum_bytes} 字节上限")
    return resolved


def distribution_module_source(root, module_name, declared_files):
    if not isinstance(module_name, str) or not module_name or not all(
        part.isidentifier() for part in module_name.split(".")
    ):
        raise ValueError("入口模块名无效")
    base = module_name.replace(".", "/")
    candidates = [f"{base}/__init__.py", f"{base}.py"]
    matches = [path for path in candidates if path in declared_files]
    if len(matches) != 1:
        raise ValueError("入口模块必须唯一映射到 distribution 内的 Python 源文件")
    relative = matches[0]
    source = safe_distribution_file(root, relative, declared_files, 8 * 1024 * 1024)
    return {
        "file": relative,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size": source.stat().st_size,
    }


def discover_components(request):
    selected_distribution = request.get("distribution")
    if selected_distribution is not None and (
        not isinstance(selected_distribution, str) or not selected_distribution.strip()
    ):
        raise UserError("COMPONENT_DISCOVERY_INVALID", "distribution 名称不能为空。")
    additional_paths = request.get("paths", [])
    if (
        not isinstance(additional_paths, list)
        or len(additional_paths) > 64
        or any(not isinstance(path, str) or len(path) > 4096 for path in additional_paths)
    ):
        raise UserError("COMPONENT_DISCOVERY_INVALID", "额外扫描路径必须是字符串数组。")

    issues = []
    local_paths = []
    if request.get("_include_project_extensions", True):
        local_paths, local_issues = project_extension_paths()
        issues.extend(local_issues)

    search_paths = []
    for raw_path in [*sys.path, *additional_paths, *local_paths]:
        path = Path(raw_path or os.getcwd())
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            if raw_path in additional_paths:
                issues.append({"path": raw_path, "message": "扫描路径不存在或无法读取。"})
            continue
        if not resolved.is_dir():
            if raw_path in additional_paths:
                issues.append({"path": raw_path, "message": "扫描路径不是目录。"})
            continue
        if str(resolved) not in search_paths:
            search_paths.append(str(resolved))

    discovered = []
    seen = set()
    manifest_bytes_total = 0
    source_bytes_total = 0
    matched_distributions = 0
    matched_entry_points = 0
    wanted = distribution_name_key(selected_distribution) if selected_distribution else None
    for distribution in importlib.metadata.distributions(path=search_paths):
        name = distribution.metadata.get("Name", "")
        version = distribution.version
        if not name or (wanted and distribution_name_key(name) != wanted):
            continue
        matched_distributions += 1
        try:
            root = Path(distribution.locate_file("")).resolve(strict=True)
            files = distribution.files
            entry_points = [point for point in distribution.entry_points if point.group == "model_maven.components"]
            if not entry_points:
                continue
            matched_entry_points += len(entry_points)
            if not root.is_dir() or files is None:
                issues.append({"distribution": name, "message": "组件 entry point 存在，但 distribution 缺少可验证的文件清单或根目录。"})
                continue
            declared_files = {PurePosixPath(str(item)).as_posix() for item in files}
        except Exception as error:
            issues.append({"distribution": name, "message": f"无法读取包元数据：{type(error).__name__}: {error}"})
            continue

        for point in entry_points:
            if len(issues) >= 255:
                issues.append({"message": "达到单次扫描最多 256 条诊断的限制。"})
                return {"components": discovered, "issues": issues, "scanned_paths": search_paths}
            identity = (distribution_name_key(name), version, str(root), point.name, point.value)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", point.name):
                    raise ValueError("entry point 名称只能包含字母、数字、点、下划线和连字符")
                module_name = entrypoint_module(point.value)
                module_source = distribution_module_source(root, module_name, declared_files)
                manifest_path = distribution_manifest_path(module_source["file"], point.name)
                manifest_file = safe_distribution_file(root, manifest_path, declared_files, 256 * 1024)
                manifest_bytes = manifest_file.read_bytes()
                if manifest_bytes_total + len(manifest_bytes) > 4 * 1024 * 1024:
                    issues.append({"message": "达到单次扫描最多读取 4 MiB 组件 manifest 的限制。"})
                    return {"components": discovered, "issues": issues, "scanned_paths": search_paths}
                manifest_bytes_total += len(manifest_bytes)
                manifest = json.loads(manifest_bytes)
                if not isinstance(manifest, dict):
                    raise ValueError("组件 manifest 顶层必须是 JSON 对象")
                validate_manifest(manifest)
                if manifest.get("entrypoint") != point.value.strip():
                    raise ValueError("manifest.entrypoint 必须与 distribution entry point 一致")
                entry_source = module_source
                source_bytes_total += entry_source.pop("size")
                module_sources = {"entrypoint": entry_source}
                fixture = manifest.get("conformance_fixture")
                if fixture is not None:
                    fixture_module = entrypoint_module(fixture)
                    fixture_source = distribution_module_source(root, fixture_module, declared_files)
                    source_bytes_total += fixture_source.pop("size")
                    module_sources["fixture"] = fixture_source
                if source_bytes_total > 64 * 1024 * 1024:
                    issues.append({"message": "达到单次扫描最多读取 64 MiB 扩展源码的限制。"})
                    return {"components": discovered, "issues": issues, "scanned_paths": search_paths}
                package_source = {
                    "distribution": name,
                    "distribution_version": version,
                    "entry_point": point.name,
                    "manifest_file": manifest_path,
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "distribution_root": str(root),
                    "module_sources": module_sources,
                }
                discovered.append({"manifest": manifest, "package_source": package_source})
            except Exception as error:
                issues.append({
                    "distribution": name,
                    "distribution_version": version,
                    "entry_point": point.name,
                    "message": f"{type(error).__name__}: {error}",
                })
            if len(discovered) >= 128:
                issues.append({"message": "达到单次扫描最多 128 个组件的限制。"})
                return {"components": discovered, "issues": issues, "scanned_paths": search_paths}
    if wanted and matched_distributions == 0:
        issues.append({"distribution":selected_distribution,"message":"当前 Python 环境和附加路径中没有找到该 distribution。"})
    elif wanted and matched_entry_points == 0:
        issues.append({"distribution":selected_distribution,"message":"该 distribution 没有 model_maven.components entry point。"})
    return {"components": discovered, "issues": issues, "scanned_paths": search_paths}


def project_extension_paths():
    project = Path.cwd().resolve(strict=True)
    metadata = project / ".model-maven"
    extension_root = metadata / "extensions"
    for path, label in ((metadata, "项目元数据"), (extension_root, "项目扩展存储")):
        if path.is_symlink():
            return [], [{"path": str(path), "message": f"{label}路径不能是符号链接。"}]
        if path.exists() and not path.is_dir():
            return [], [{"path": str(path), "message": f"{label}路径不是目录。"}]
    if not extension_root.is_dir():
        return [], []
    paths = []
    issues = []
    try:
        children = sorted(extension_root.iterdir(), key=lambda item: item.name)
    except OSError as error:
        return [], [{"path": str(extension_root), "message": f"无法扫描项目扩展存储：{error}"}]
    for path in children:
        if not re.fullmatch(r"[a-f0-9]{64}", path.name):
            continue
        if path.is_symlink() or not path.is_dir():
            issues.append({"path": str(path), "message": "扩展包目录必须是普通目录。"})
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(extension_root.resolve(strict=True))
        except (OSError, ValueError):
            issues.append({"path": str(path), "message": "扩展包目录离开项目扩展存储。"})
            continue
        paths.append(str(resolved))
    return paths, issues


