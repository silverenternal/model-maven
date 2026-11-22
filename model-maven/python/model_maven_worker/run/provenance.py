"""Conservative project-source snapshots used by runs and model variants."""

import hashlib
import sys
from pathlib import Path
from typing import Iterable

from ..protocol import UserError


def source_manifest(
    root,
    loaded_modules: Iterable[object] | None = None,
    code_root=None,
):
    """Hash loaded local modules and their package trees conservatively.

    This does not claim to capture data, external weights or future dynamic imports.
    """
    root = Path(root).resolve()
    code_root = Path(code_root).resolve() if code_root is not None else None
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
        ignored = {".venv", "venv", "target", ".git", "__pycache__"}
        if (
            path.suffix == ".py"
            and path.is_file()
            and not any(part in ignored for part in relative.parts)
        ):
            files[manifest_path] = source_path

    for module in sys.modules.values() if loaded_modules is None else loaded_modules:
        filename = getattr(module, "__file__", None)
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
            manifest[str(path)] = hashlib.sha256(source_path.read_bytes()).hexdigest()
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
            matches = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
        except OSError:
            matches = False
        if not matches:
            raise UserError(
                "SOURCE_CHANGED", f"基础源码已变化：{filename}",
                "旧变体不会自动重放到新代码。请保留旧描述并用新名称重新建立变体。",
            )
