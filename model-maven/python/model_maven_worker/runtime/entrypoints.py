"""Resolve the public module:callable reference used by Python project inputs."""

import importlib
import sys
from pathlib import Path

from ..protocol import UserError


def configure_project_import_roots(request):
    """Apply explicitly selected project-relative import roots to this worker.

    Snapshot-backed runs resolve the same relative roots inside the snapshot;
    they must never substitute the live project directory when a root is absent.
    """
    runtime = request.get("runtime")
    roots = runtime.get("project_import_roots") if isinstance(runtime, dict) else None
    if roots is None or roots == []:
        return []
    if not isinstance(roots, list) or len(roots) > 32:
        raise UserError("IMPORT_ROOTS_INVALID", "项目模块搜索目录必须是最多 32 项的相对路径数组。")
    try:
        project = Path(request["project_root"]).resolve(strict=True)
        code = Path(request.get("code_snapshot_root") or project).resolve(strict=True)
    except (KeyError, TypeError, OSError, RuntimeError) as error:
        raise UserError("IMPORT_ROOTS_INVALID", "无法确定项目或源码快照目录。") from error
    selected = []
    for value in roots:
        if (not isinstance(value, str) or not value or len(value) > 1024
                or "\x00" in value or "\\" in value):
            raise UserError("IMPORT_ROOTS_INVALID", "模块搜索目录必须是有效的项目内相对路径。")
        relative = Path(value)
        if relative.is_absolute() or any(part in (".", "..") for part in value.split("/")):
            raise UserError("IMPORT_ROOTS_INVALID", "模块搜索目录不能是绝对路径或包含目录跳转。")
        try:
            selected_path = (code / relative).resolve(strict=True)
            selected_path.relative_to(code)
        except (OSError, ValueError, RuntimeError) as error:
            raise UserError("IMPORT_ROOT_UNAVAILABLE", f"模块搜索目录不存在或超出项目／源码快照：{value}",
                            "检查目录或重新创建源码快照；不会自动切换到实时源码。") from error
        if not selected_path.is_dir():
            raise UserError("IMPORT_ROOT_UNAVAILABLE", f"模块搜索路径不是目录：{value}")
        entry = str(selected_path)
        if entry not in selected:
            selected.append(entry)
    for entry in reversed(selected):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    return selected


def entrypoint(value):
    if not isinstance(value, str) or value.count(":") != 1:
        raise UserError("ENTRYPOINT_INVALID", "入口应为 package.module:callable。")
    module_name, attribute = value.split(":")
    if not all(part.isidentifier() for part in module_name.split(".")) or not all(
        part.isidentifier() for part in attribute.split(".")
    ):
        raise UserError("ENTRYPOINT_INVALID", "入口只能包含模块/属性名称，不能包含表达式。")
    try:
        module = importlib.import_module(module_name)
        target = module
        for part in attribute.split("."):
            target = getattr(target, part)
    except Exception as error:
        raise UserError("ENTRYPOINT_LOAD_FAILED", f"无法加载 {value}: {error}") from error
    if not callable(target):
        raise UserError("ENTRYPOINT_NOT_CALLABLE", f"{value} 不是可调用对象。")
    return target
