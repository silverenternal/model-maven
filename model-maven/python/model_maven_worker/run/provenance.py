"""Conservative project-source snapshots used by runs and model variants."""

import sys
from pathlib import Path
from typing import Iterable

from ..protocol import UserError
from .file_digest import sha256_file


def _module_file(module):
    # Dynamic namespaces can synthesize __file__ despite having no source file.
    # Read the metadata recorded on the module without invoking __getattr__.
    namespace = getattr(module, "__dict__", None)
    return namespace.get("__file__") if isinstance(namespace, dict) else None


def _is_selected(relative, import_roots):
    for item in import_roots or []:
        if isinstance(item, str):
            candidate = Path(item)
            if not candidate.is_absolute() and ".." not in candidate.parts and (relative == candidate or candidate in relative.parents):
                return True
    return False


def check_snapshot_project_imports(root, code_root, loaded_modules=None, import_roots=None):
    """Check currently loaded module files without rescanning package trees."""
    if code_root is None:
        return
    project = Path(root).resolve()
    snapshot = Path(code_root).resolve()
    environment = Path(sys.prefix).resolve()
    for module in list(sys.modules.values()) if loaded_modules is None else loaded_modules:
        filename = _module_file(module)
        if not isinstance(filename, (str, bytes, Path)):
            continue
        try:
            path = Path(filename).resolve()
            if path.suffix not in {".py", ".pyc"}:
                continue
            if path.is_relative_to(snapshot):
                continue
            relative = path.relative_to(project)
            if not _is_selected(relative, import_roots) and (
                environment != project and environment.is_relative_to(project)
                and path.is_relative_to(environment)
            ):
                continue
        except (OSError, ValueError, TypeError):
            continue
        raise UserError(
            "SOURCE_SNAPSHOT_LIVE_IMPORT",
            f"快照运行观察到项目实时源码导入：{relative.as_posix()}。",
            "检查 PYTHONPATH、可编辑安装或自定义导入；该模块未作为固定源码运行。",
            {"relative_path": relative.as_posix(), "scope": "observed-module-file"},
        )


def source_manifest(
    root,
    loaded_modules: Iterable[object] | None = None,
    code_root=None,
    import_roots=None,
):
    """Hash loaded local modules and their package trees conservatively.

    This does not claim to capture data, external weights or future dynamic imports.
    """
    root = Path(root).resolve()
    code_root = Path(code_root).resolve() if code_root is not None else None
    environment = Path(sys.prefix).resolve()
    files = {}

    def include(candidate):
        try:
            path = Path(candidate).resolve()
            if code_root is not None:
                try:
                    relative = path.relative_to(code_root)
                    source_path = path
                    manifest_path = root / relative
                except ValueError:
                    relative = path.relative_to(root)
                    source_path = path
                    manifest_path = path
            else:
                relative = path.relative_to(root)
                source_path = path
                manifest_path = path
        except (OSError, ValueError, TypeError):
            return
        # A project may legitimately name a package "target" or "venv".
        # Exclude the actual interpreter environment, not similarly named code.
        # Explicit import roots remain project source even inside that environment.
        if (code_root is None and environment != root and environment.is_relative_to(root)
                and path.is_relative_to(environment) and not _is_selected(relative, import_roots)):
            return
        if (
            path.suffix == ".py"
            and path.is_file()
            and not any(part in {".git", "__pycache__"} for part in relative.parts)
        ):
            files[manifest_path] = source_path

    modules = list(sys.modules.values()) if loaded_modules is None else list(loaded_modules)
    check_snapshot_project_imports(root, code_root, modules, import_roots)
    for module in modules:
        filename = _module_file(module)
        if not filename:
            continue
        try:
            path = Path(filename).resolve()
        except (OSError, TypeError, ValueError):
            continue
        before = len(files)
        include(path)
        if len(files) == before:
            continue
        package = path.parent
        if (package / "__init__.py").is_file():
            for candidate in package.rglob("*.py"):
                include(candidate)
    manifest = {}
    for path, source_path in sorted(files.items()):
        try:
            manifest[str(path)] = sha256_file(source_path)
        except OSError as error:
            raise UserError(
                "SOURCE_SNAPSHOT_FAILED",
                f"无法固定项目源码 {source_path}: {error}",
                "检查源码是否在编辑期间被移动或删除；原变体不会被覆盖。",
            ) from error
    return manifest


def verify_manifest(manifest):
    if not isinstance(manifest, dict):
        raise UserError("SOURCE_MANIFEST_INVALID", "源码指纹清单必须是对象。")
    for filename, expected in manifest.items():
        path = Path(filename)
        try:
            matches = path.is_file() and sha256_file(path) == expected
        except OSError:
            matches = False
        if not matches:
            raise UserError(
                "SOURCE_CHANGED", f"基础源码已变化：{filename}",
                "旧变体不会自动重放到新代码。请保留旧描述并用新名称重新建立变体。",
            )
