"""Layer construction and native container edit operations."""

from dataclasses import dataclass
from typing import Any, Callable

from ..protocol import UserError
from ..model.structure import registered_storage_key
from .providers import apply_provider
from .sharing import clone_selected, share_selected
from .initialization_rng import layer_rng_scope


@dataclass(frozen=True)
class LayerEditRuntime:
    torch: Any
    entrypoint: Callable[[str], Any]
    resolve_module: Callable[[Any, str], Any]
    module_entries: Callable[[Any], tuple]
    normalize_path: Callable[[str], str]
    editable_params: Callable[[Any], dict | None]


def old_state_names(module):
    """Return observable state names, or None if a custom module cannot report them."""
    try:
        return list(module.state_dict())
    except Exception:
        return None


def new_layer(runtime, edit, *, preserve_factory_device=False):
    if edit.get("component"):
        target = runtime.entrypoint(edit["component"])
    else:
        name = edit.get("layer_type")
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise UserError("LAYER_TYPE_UNSUPPORTED", "Layer 类型必须是 torch.nn 中的公开类名。",
                            "自定义层可用 --component module:factory。")
        target = getattr(runtime.torch.nn, name, None)
        if not isinstance(target, type) or not issubclass(target, runtime.torch.nn.Module):
            raise UserError("LAYER_TYPE_UNSUPPORTED", f"当前 Python 环境的 torch.nn 中没有可用的 Module 类 {name!r}。",
                            "检查所选 PyTorch 版本，或使用 --component module:factory。")
    try:
        initialization_seed = edit.get("initialization_seed")
        if type(initialization_seed) is int and 0 <= initialization_seed < 2**64:
            with layer_rng_scope(runtime.torch, initialization_seed, edit.get("rng_devices", [])):
                layer = target(**edit.get("params", {}))
        else:
            if "rng_devices" in edit:
                raise UserError("LAYER_RNG_SEED_REQUIRED", "指定随机设备时必须同时固定 Layer 初始化种子。")
            layer = target(**edit.get("params", {}))
    except UserError:
        raise
    except Exception as error:
        raise UserError("LAYER_CONSTRUCTION_FAILED", str(error)) from error
    if not isinstance(layer, runtime.torch.nn.Module):
        raise UserError("LAYER_TYPE_INVALID", "Layer 工厂必须返回 nn.Module。")
    return layer if preserve_factory_device else layer.cpu()


def layer_container_kind(runtime, module):
    if isinstance(module, runtime.torch.nn.Sequential):
        return "Sequential"
    if isinstance(module, runtime.torch.nn.ModuleList):
        return "ModuleList"
    if isinstance(module, runtime.torch.nn.ModuleDict):
        return "ModuleDict"
    return None


def _validate_layer_name(parent, name):
    """Match nn.Module.add_module's registry-name constraints."""
    if not isinstance(name, str) or not name or "." in name or hasattr(parent, name):
        raise UserError(
            "LAYER_NAME_INVALID",
            "新层名称必须是非空字符串，不能包含点号，也不能覆盖容器已有属性。",
        )


def _automatic_sequential_name(parent):
    """Use Sequential's conventional decimal child names without collisions."""
    index = len(parent._modules)
    while str(index) in parent._modules or hasattr(parent, str(index)):
        index += 1
    return str(index)


def replace_module(runtime, model, path, replacement):
    old = runtime.resolve_module(model, path)
    if old is model:
        raise UserError("ROOT_EDIT_UNSUPPORTED", "当前不替换整个根模型，请指定子层。")
    modules, _ = runtime.module_entries(model)
    changed = []
    # Rebind all registered aliases to preserve physical sharing.
    for parent in modules.values():
        for name, child in list(parent._modules.items()):
            if child is old:
                parent._modules[name] = replacement
                changed.append(name)
    return old, changed



def _external_shared_state(model, old, aliases):
    """Find registered object or inspectable storage aliases outside the layer."""
    result = []
    aliases = [alias if alias == "model" or alias.startswith("model.") else "model." + alias for alias in aliases]
    external_objects, external_storages = {}, {}
    for _, accessor in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
        for path, tensor in getattr(model, accessor)(remove_duplicate=False):
            full_path = "model." + path
            if any(full_path.startswith(alias + ".") for alias in aliases):
                continue
            external_objects.setdefault(id(tensor), set()).add(full_path)
            key = registered_storage_key(tensor)
            if key is not None:
                external_storages.setdefault(key, []).append((id(tensor), full_path))
    for kind, accessor in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
        for name, tensor in getattr(old, accessor)(remove_duplicate=False):
            paths = set(external_objects.get(id(tensor), ()))
            key = registered_storage_key(tensor)
            storage_paths = {path for other_id, path in external_storages.get(key, ()) if other_id != id(tensor)} if key is not None else set()
            paths.update(storage_paths)
            if paths:
                item = {"kind": kind, "name": name, "tensor": tensor, "external_paths": sorted(paths)}
                if storage_paths:
                    item["sharing"] = "storage"
                result.append(item)
    return result


def unverified_state_storage(module):
    """Registered target states whose storage aliases cannot be inspected."""
    return [
        {"kind": kind, "name": name}
        for kind, accessor in (("parameter", "named_parameters"), ("buffer", "named_buffers"))
        for name, tensor in getattr(module, accessor)(remove_duplicate=False)
        if registered_storage_key(tensor) is None
    ]


def _preserve_external_shared_state(replacement, shared, compatible):
    for item in shared:
        name = item["name"]
        parent_path, _, local_name = name.rpartition(".")
        try:
            parent = replacement.get_submodule(parent_path) if parent_path else replacement
        except AttributeError:
            parent = None
        registry = None if parent is None else (parent._parameters if item["kind"] == "parameter" else parent._buffers)
        target = registry.get(local_name) if registry is not None else None
        if not compatible or target is None or target.shape != item["tensor"].shape or target.dtype != item["tensor"].dtype:
            raise UserError(
                "SHARED_STATE_EDIT_UNSUPPORTED",
                f"此操作会切断共享{item['kind']} {name} 与 {', '.join(item['external_paths'])} 的关联。",
                "请先显式解除共享，或使用自定义编辑实现共享状态及 checkpoint 迁移。",
                {"state": name, "kind": item["kind"], "external_paths": item["external_paths"]},
            )
    # Validate the entire mapping before touching the candidate.
    for item in shared:
        parent_path, _, local_name = item["name"].rpartition(".")
        parent = replacement.get_submodule(parent_path) if parent_path else replacement
        registry = parent._parameters if item["kind"] == "parameter" else parent._buffers
        registry[local_name] = item["tensor"]


def apply_edit(runtime, model, edit, *, preserve_factory_device=False):
    action = edit.get("action")
    if action == "custom":
        return apply_provider(runtime, model, edit)
    if action == "sharing":
        mode = edit.get("mode")
        if mode == "clone-selected":
            return clone_selected(runtime, model, edit.get("path"))
        if mode == "share-selected":
            return share_selected(runtime, model, edit.get("source"), edit.get("targets"))
        raise UserError("SHARING_PLAN_INVALID", "sharing mode 必须是 clone-selected 或 share-selected。")

    params = edit.get("params", {})
    if not isinstance(params, dict):
        raise UserError("EDIT_INVALID", "参数必须是 JSON 对象。")
    effect = {"action": action, "path": edit.get("path", edit.get("parent")),
              "preserved": [], "initialized": []}
    if action == "add":
        parent = runtime.resolve_module(model, edit["parent"])
        container_type = layer_container_kind(runtime, parent)
        if container_type is None:
            raise UserError("CONTAINER_UNSUPPORTED", "自动插入仅支持 nn.Sequential、nn.ModuleList、nn.ModuleDict 及其保留原生 _modules 注册语义的子类。", "其他自定义分支请使用 Layer Edit Provider；不会猜测 forward 接线。")
        name = edit.get("name")
        before, after = edit.get("before"), edit.get("after")
        anchor = before or after
        if before and after:
            raise UserError("POSITION_INVALID", "before 与 after 不能同时使用。")
        position_before, position_after = before, after
        if container_type == "ModuleList":
            if name not in (None, ""):
                raise UserError("LAYER_NAME_UNSUPPORTED", "ModuleList 使用数字索引命名；添加时省略 name。")
            if anchor:
                anchor = str(anchor).rsplit(".", 1)[-1]
                if anchor not in parent._modules:
                    raise UserError("ANCHOR_NOT_FOUND", f"ModuleList 中找不到索引：{anchor}")
                position_before = anchor if before else None
                position_after = anchor if after else None
        elif container_type == "ModuleDict":
            _validate_layer_name(parent, name)
            if anchor:
                anchor = str(anchor).rsplit(".", 1)[-1]
                if anchor not in parent._modules:
                    raise UserError("ANCHOR_NOT_FOUND", f"ModuleDict 中找不到分支：{anchor}")
                position_before = anchor if before else None
                position_after = anchor if after else None
        else:
            if name in (None, ""):
                name = _automatic_sequential_name(parent)
            _validate_layer_name(parent, name)
            if anchor and anchor not in parent._modules:
                raise UserError("ANCHOR_NOT_FOUND", f"容器中找不到位置：{anchor}")
        layer = new_layer(runtime, edit, preserve_factory_device=preserve_factory_device)
        effect["placement_policy"] = "factory" if preserve_factory_device else "legacy_cpu"
        if container_type == "ModuleList":
            rebuilt = list(parent._modules.items())
            insertion_index = next((i for i, (key, _) in enumerate(rebuilt) if key == anchor), len(rebuilt))
            if after and anchor:
                insertion_index += 1
            rebuilt.insert(insertion_index, ("", layer))
            rebuilt = [(str(index), child) for index, (_, child) in enumerate(rebuilt)]
            name = str(insertion_index)
        else:
            rebuilt = []
            for key, child in parent._modules.items():
                if position_before == key:
                    rebuilt.append((name, layer))
                rebuilt.append((key, child))
                if position_after == key:
                    rebuilt.append((name, layer))
            if not anchor:
                rebuilt.append((name, layer))
        parent._modules.clear()
        parent._modules.update(rebuilt)
        effect["name"] = name
        effect["module_type"] = type(layer).__name__
        effect["initialized"] = list(layer.state_dict())
        return effect
    if action not in {"set", "replace", "remove"}:
        raise UserError("EDIT_INVALID", f"未知编辑操作：{action}")
    path = runtime.normalize_path(edit["path"])
    old = runtime.resolve_module(model, path)
    if old is model:
        raise UserError("ROOT_EDIT_UNSUPPORTED", "不能通过 Layer 操作删除或重建根模型。")
    unverified = unverified_state_storage(old)
    if unverified:
        effect["unverified_old_state_storage"] = unverified
    if action == "remove" and edit.get("strategy", "remove") == "remove":
        parent_path, name = path.rsplit(".", 1)
        parent = runtime.resolve_module(model, parent_path)
        container_type = layer_container_kind(runtime, parent)
        if container_type is None:
            raise UserError("CONTAINER_UNSUPPORTED", "真正删除仅支持 Sequential/ModuleList/ModuleDict 及其保留原生 _modules 注册语义的子类。", "--strategy bypass 可明确用 Identity 替换单输入层；仍会验证前向。")
        _, aliases = runtime.module_entries(model)
        if len(aliases[id(old)]) > 1:
            raise UserError("SHARED_REMOVE_UNSUPPORTED", "该层有共享别名，当前不猜测删除哪条引用。")
        effect["old_state_not_copied"] = old_state_names(old)
        del parent._modules[name]
        if container_type == "ModuleList":
            remaining = list(parent._modules.values())
            parent._modules.clear()
            parent._modules.update((str(index), child) for index, child in enumerate(remaining))
        return effect
    if action == "set":
        known = runtime.editable_params(old)
        if known is None:
            raise UserError("FIELD_EDIT_UNSUPPORTED", "该原生层没有已实现的参数重建处理器。", "使用 replace --component module:factory。")
        if not params or set(params) - set(known):
            raise UserError("FIELD_INVALID", f"可编辑字段：{', '.join(known)}")
        values = dict(known)
        values.update(params)
        replacement = new_layer(runtime, {"layer_type": type(old).__name__, "params": values}, preserve_factory_device=preserve_factory_device)
    elif action == "replace":
        replacement = new_layer(runtime, edit, preserve_factory_device=preserve_factory_device)
    else:
        if edit.get("strategy") != "bypass":
            raise UserError("STRATEGY_INVALID", "策略应为 remove 或 bypass。")
        replacement = runtime.torch.nn.Identity()
    _, aliases = runtime.module_entries(model)
    affected_aliases = aliases[id(old)]
    shared_state = _external_shared_state(model, old, affected_aliases)
    old_state = next(old.parameters(), None)
    if old_state is None:
        old_state = next(old.buffers(), None)
    if old_state is not None:
        effect["placement_policy"] = "old_layer_first_registered_state"
        effect["placement_reference"] = {"device": str(old_state.device), "dtype": str(old_state.dtype),
                                         "dtype_applied": bool(old_state.is_floating_point() or old_state.is_complex())}
        if old_state.is_floating_point() or old_state.is_complex():
            replacement.to(device=old_state.device, dtype=old_state.dtype)
        else:
            replacement.to(device=old_state.device)
    else:
        effect["placement_policy"] = "factory" if preserve_factory_device else "legacy_cpu"
    replacement.train(old.training)
    # Only built-in same-class reconstruction has an implicit state mapping.
    state = replacement.state_dict()
    previous = old.state_dict()
    compatible = action == "set" and type(old) is type(replacement)
    for name, tensor in state.items():
        if compatible and name in previous and tensor.shape == previous[name].shape:
            tensor.copy_(previous[name])
            effect["preserved"].append(name)
        else:
            effect["initialized"].append(name)
    effect["old_state_not_copied"] = [name for name in previous if name not in effect["preserved"]]
    replacement.load_state_dict(state)
    _preserve_external_shared_state(replacement, shared_state, compatible)
    effect["affected_aliases"] = affected_aliases
    if shared_state:
        effect["preserved_external_state_sharing"] = [
            {key: value for key, value in item.items() if key != "tensor"}
            for item in shared_state
        ]
    replace_module(runtime, model, path, replacement)
    return effect


def apply_history_item(runtime, model, item, *, preserve_factory_device=False):
    if item.get("action") != "transaction":
        return [apply_edit(runtime, model, item, preserve_factory_device=preserve_factory_device)]
    operations = item.get("operations")
    if not isinstance(operations, list) or not operations:
        raise UserError("EDIT_PLAN_INVALID", "原子 Layer 计划必须包含至少一项操作。")
    return [apply_edit(runtime, model, operation, preserve_factory_device=preserve_factory_device) for operation in operations]
