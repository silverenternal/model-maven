"""Explicit, path-based helpers for preserving or changing module sharing."""

import copy

from ..protocol import UserError


def module_paths(runtime, model):
    """Return stable registered module paths; never use dynamic call numbers."""
    paths = {}

    def visit(module, prefix, ancestors):
        if id(module) in ancestors:
            return
        for name, child in module._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            paths.setdefault(id(child), []).append(path)
            visit(child, path, ancestors | {id(module)})

    paths.setdefault(id(model), []).append("model")
    visit(model, "model", set())
    return paths


def aliases_for(runtime, model, target):
    module = runtime.resolve_module(model, target)
    paths = module_paths(runtime, model).get(id(module), [])
    return module, tuple(sorted(paths))


def _parent_and_name(runtime, model, path):
    normalized = runtime.normalize_path(path)
    if normalized in {"", "model"}:
        raise UserError("ROOT_EDIT_UNSUPPORTED", "不能更改根模型的共享身份，请指定一个注册子模块路径。")
    parent_path, _, name = normalized.rpartition(".")
    parent = runtime.resolve_module(model, parent_path or "model")
    if name not in parent._modules or parent._modules[name] is None:
        raise UserError("MODULE_PATH_INVALID", f"路径 {normalized} 不是已注册的直接子模块。")
    return normalized, parent, name


def clone_selected(runtime, model, path):
    """Break one alias by copying its module and all tensor state."""
    normalized, parent, name = _parent_and_name(runtime, model, path)
    old = parent._modules[name]
    old_aliases = tuple(sorted(module_paths(runtime, model).get(id(old), ())))
    if len(old_aliases) < 2:
        raise UserError("MODULE_NOT_SHARED", f"{normalized} 没有其它共享别名，无需解除共享。")
    try:
        cloned = copy.deepcopy(old)
    except Exception as error:
        raise UserError("MODULE_CLONE_FAILED", f"无法复制共享模块 {normalized}：{type(error).__name__}: {error}") from error
    parent._modules[name] = cloned
    retained = tuple(sorted(module_paths(runtime, model).get(id(old), ())))
    return {
        "action": "clone-selected",
        "selected_path": normalized,
        "retained_aliases": list(retained),
        "cloned_aliases": [normalized],
        "state_policy": "deepcopy_module_and_state",
    }


def share_selected(runtime, model, source, targets):
    """Bind explicitly named target paths to the source module instance."""
    if not isinstance(targets, list) or not targets or len(targets) > 64:
        raise UserError("SHARING_PLAN_INVALID", "share-selected 需要 1..64 个显式目标模块路径。")
    source_path, source_parent, source_name = _parent_and_name(runtime, model, source)
    source_module = source_parent._modules[source_name]
    changed = []
    discarded = []
    for target in targets:
        target_path, parent, name = _parent_and_name(runtime, model, target)
        if target_path == source_path:
            raise UserError("SHARING_PLAN_INVALID", "共享目标不能与源路径相同。")
        if target_path.startswith(source_path + ".") or source_path.startswith(target_path + "."):
            raise UserError("SHARING_PLAN_INVALID", "不能将父模块与其子模块重绑，否则会创建循环模型结构。")
        if parent._modules[name] is source_module:
            raise UserError("SHARING_PLAN_INVALID", f"{target_path} 已与 {source_path} 共享同一模块。")
        previous = parent._modules[name]
        discarded.append({
            "path": target_path,
            "module_type": type(previous).__name__,
            "state_keys": sorted(previous.state_dict()),
            "state_policy": "replaced_by_source_module_state",
        })
        parent._modules[name] = source_module
        changed.append(target_path)
    aliases = tuple(sorted(module_paths(runtime, model).get(id(source_module), ())))
    return {
        "action": "share-selected",
        "source_path": source_path,
        "target_paths": sorted(changed),
        "shared_aliases": list(aliases),
        "replaced_target_state": discarded,
        "state_policy": "target_modules_replaced_by_source_instance_and_state",
    }
