"""Read-only inspection helpers for installed extension packages and references."""

import ast
import importlib.metadata
import json
import os
import re
from pathlib import Path

from ..protocol import UserError


def component_json_files(directory, limit_bytes=32 * 1024 * 1024):
    if directory.is_symlink():
        raise UserError("COMPONENT_STORE_UNSAFE", f"组件数据目录不安全：{directory}")
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise UserError("COMPONENT_STORE_UNSAFE", f"组件数据目录不安全：{directory}")
    visited_bytes = 0
    for current, directories, files in os.walk(directory, topdown=True, followlinks=False):
        parent = Path(current)
        for name in list(directories):
            child = parent / name
            if child.is_symlink():
                raise UserError("COMPONENT_STORE_UNSAFE", f"组件数据目录含符号链接：{child}")
        directories.sort()
        for name in sorted(files):
            path = parent / name
            if path.suffix != ".json":
                continue
            if path.is_symlink() or not path.is_file():
                raise UserError("COMPONENT_STORE_UNSAFE", f"组件数据文件不是普通文件：{path}")
            size = path.stat().st_size
            visited_bytes += size
            if size > limit_bytes or visited_bytes > 128 * 1024 * 1024:
                raise UserError("COMPONENT_STORE_TOO_LARGE", f"组件数据扫描超过限制：{path}")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise UserError("COMPONENT_STORE_UNREADABLE", f"无法安全检查组件数据文件 {path}：{error}") from error
            yield path, value


def component_reference_paths(value, component_id, version, path):
    matches = []
    if isinstance(value, dict):
        reference = value.get("component_ref")
        if (
            isinstance(reference, dict)
            and reference.get("id") == component_id
            and reference.get("version") == version
        ):
            matches.append({"path": str(path), "component": f"{component_id}@{version}"})
        for child in value.values():
            matches.extend(component_reference_paths(child, component_id, version, path))
    elif isinstance(value, list):
        for child in value:
            matches.extend(component_reference_paths(child, component_id, version, path))
    return matches


def component_source_imports(project_root, manifest, extension_root, selected_environment_venv, safe_project_path):
    module_roots = set()
    for key in ("entrypoint", "conformance_fixture"):
        value = manifest.get(key)
        if isinstance(value, str) and ":" in value:
            module = value.split(":", 1)[0].strip()
            if module and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
                module_roots.add(module.split(".", 1)[0])
    package_source = manifest.get("package_source")
    raw_root = package_source.get("distribution_root") if isinstance(package_source, dict) else None
    if isinstance(raw_root, str):
        try:
            distribution_root = Path(raw_root).resolve(strict=True)
            distributions = importlib.metadata.distributions(path=[str(distribution_root)])
            for distribution in distributions:
                top_level = distribution.read_text("top_level.txt")
                if top_level:
                    module_roots.update(
                        line.strip()
                        for line in top_level.splitlines()
                        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", line.strip())
                    )
            for child in distribution_root.iterdir():
                if child.name.endswith((".dist-info", ".data")):
                    continue
                if child.is_dir() and any(child.rglob("*.py")):
                    module_roots.add(child.name)
                elif child.is_file() and child.suffix == ".py":
                    module_roots.add(child.stem)
        except (OSError, ValueError):
            pass
    if not module_roots:
        return [], []

    ignored = {".git", ".model-maven", ".venv", "venv", "target", "build", "dist", "node_modules", "__pycache__"}
    imports = []
    unreadable = []
    ignored_roots = []
    try:
        selected_venv = selected_environment_venv()
        ignored_roots.append(
            safe_project_path(selected_venv, "项目虚拟环境")
            .relative_to(project_root)
        )
    except UserError as error:
        unreadable.append({"path": ".model-maven/environment.json", "reason": str(error)})
    files_seen = 0
    for current, directories, files in os.walk(project_root, topdown=True, followlinks=False):
        parent = Path(current)
        retained_directories = []
        for name in directories:
            child = parent / name
            if name in ignored:
                continue
            relative_child = child.relative_to(project_root)
            if any(relative_child == root or root in relative_child.parents for root in ignored_roots):
                continue
            if child.is_symlink():
                unreadable.append({"path": child.relative_to(project_root).as_posix(), "reason": "符号链接源码目录未扫描"})
                continue
            retained_directories.append(name)
        directories[:] = sorted(retained_directories)
        if extension_root is not None and (parent == extension_root or extension_root in parent.parents):
            directories.clear()
            continue
        for name in sorted(files):
            path = parent / name
            if path.suffix != ".py":
                continue
            if path.is_symlink():
                unreadable.append({"path": path.relative_to(project_root).as_posix(), "reason": "符号链接 Python 文件未扫描"})
                continue
            if not path.is_file():
                continue
            files_seen += 1
            if files_seen > 20000:
                unreadable.append({"path": str(path.relative_to(project_root)), "reason": "Python 源码扫描达到 20000 个文件上限"})
                return imports, unreadable
            try:
                if path.stat().st_size > 8 * 1024 * 1024:
                    unreadable.append({"path": str(path.relative_to(project_root)), "reason": "文件超过 8 MiB 扫描上限"})
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, UnicodeDecodeError, SyntaxError) as error:
                unreadable.append({"path": str(path.relative_to(project_root)), "reason": f"无法解析源码：{type(error).__name__}"})
                continue
            candidates = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    candidates.extend((item.name, node.lineno) for item in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    candidates.append((node.module, node.lineno))
                elif isinstance(node, ast.Call):
                    function = node.func
                    function_name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ""
                    if function_name in {"import_module", "__import__"} and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        candidates.append((node.args[0].value, node.lineno))
            relative = path.relative_to(project_root).as_posix()
            for module, line in candidates:
                if module.split(".", 1)[0] in module_roots:
                    imports.append({"path": relative, "line": line, "module": module})
    imports.sort(key=lambda item: (item["path"], item["line"], item["module"]))
    unreadable.sort(key=lambda item: item["path"])
    return imports, unreadable

