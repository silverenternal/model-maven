"""Resolve the public module:callable reference used by Python project inputs."""

import importlib

from ..protocol import UserError


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
