"""Readable Python generation for explicit root-factory assembly recipes.

This intentionally generates a small adapter around user-owned factories. It
never rewrites the user's model source and does not claim to package runtime or
third-party dependencies; component lock data is emitted as provenance comments.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from ..protocol import UserError
from .initialization_rng import validate_rng_devices


_MAX_RECIPE_BYTES = 1024 * 1024
def _entrypoint(value: Any, path: str) -> str:
    if not isinstance(value, str) or value.count(":") != 1:
        raise UserError("ASSEMBLY_RECIPE_INVALID", f"{path} 必须是 module:callable 入口。")
    module, attribute = value.split(":", 1)
    if not all(part.isidentifier() for part in module.split(".")) or not all(
        part.isidentifier() for part in attribute.split(".")
    ):
        raise UserError("ASSEMBLY_RECIPE_INVALID", f"{path} 入口格式无效：{value!r}。")
    return value


def _json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise UserError("ASSEMBLY_RECIPE_INVALID", f"{path} 不能包含 NaN 或无穷值。")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item, f"{path}.{key}") for key, item in value.items()}
    raise UserError("ASSEMBLY_RECIPE_INVALID", f"{path} 必须是 JSON 数据。")


def _parameters(value: Any, path: str) -> dict:
    if not isinstance(value, dict) or len(value) > 256:
        raise UserError("ASSEMBLY_RECIPE_INVALID", f"{path} 必须是最多 256 项的参数对象。")
    for key in value:
        if not isinstance(key, str):
            raise UserError("ASSEMBLY_RECIPE_INVALID", f"{path} 的参数名必须是字符串：{key!r}。")
    return _json_value(value, path)


def _variant_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path} 必须是有效模块路径。")
    parts = value.split(".")
    if parts[0] == "model":
        parts = parts[1:]
    if any(
        not part
        or part in {".", ".."}
        or any(ord(char) < 32 for char in part)
        for part in parts
    ):
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path} 模块路径无效。")
    return "model" + ("." + ".".join(parts) if parts else "")


def _variant_child_path(value: Any, path: str) -> str:
    normalized = _variant_path(value, path)
    if normalized == "model":
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path} 不能指向整个根模型。")
    return normalized


def _variant_layer(operation: dict, path: str) -> dict:
    component = operation.get("component")
    layer_type = operation.get("layer_type")
    component_ref = operation.get("component_ref")
    if component_ref is not None:
        raise UserError(
            "ASSEMBLY_VARIANT_COMPONENT_UNSUPPORTED",
            f"{path} 使用了注册组件；当前降低器只接受 Python module:factory 接缝。",
        )
    if component is not None:
        component = _entrypoint(component, f"{path}.component")
        layer_type = None
    elif not isinstance(layer_type, str) or not layer_type.isidentifier() or layer_type.startswith("_"):
        raise UserError(
            "ASSEMBLY_VARIANT_LAYER_UNSUPPORTED",
            f"{path}.layer_type 必须是 torch.nn 中的公开类名；实际可用性由运行时验证。",
        )
    return {
        "component": component,
        "layer_type": layer_type,
        "params": _parameters(operation.get("params", {}), f"{path}.params"),
    }


def _variant_operation(operation: Any, path: str, depth: int = 0) -> dict:
    if depth > 2 or not isinstance(operation, dict):
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path} 操作格式无效。")
    action = operation.get("action")
    if "rng_devices" in operation and action not in {"add", "replace"}:
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path}.rng_devices 仅适用于 add/replace。")
    if action == "transaction":
        operations = operation.get("operations")
        if not isinstance(operations, list) or not 1 <= len(operations) <= 64:
            raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path} 事务必须包含 1..64 项操作。")
        return {
            "action": "transaction",
            "operations": [
                _variant_operation(item, f"{path}.operations[{index}]", depth + 1)
                for index, item in enumerate(operations)
            ],
        }
    if action == "custom":
        raise UserError(
            "ASSEMBLY_VARIANT_PROVIDER_UNSUPPORTED",
            f"{path} 调用了自定义 Layer Edit Provider；代码生成器不会猜测其图语义。",
        )
    if action == "sharing":
        mode = operation.get("mode")
        if mode == "clone-selected":
            return {"action": "sharing", "mode": mode, "path": _variant_child_path(operation.get("path"), f"{path}.path")}
        targets = operation.get("targets")
        if (
            mode != "share-selected"
            or not isinstance(targets, list)
            or not 1 <= len(targets) <= 64
        ):
            raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path} sharing 操作格式无效。")
        return {
            "action": "sharing",
            "mode": mode,
            "source": _variant_child_path(operation.get("source"), f"{path}.source"),
            "targets": [_variant_child_path(item, f"{path}.targets[{index}]") for index, item in enumerate(targets)],
        }
    if action == "add":
        before, after = operation.get("before"), operation.get("after")
        if before is not None and after is not None:
            raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path} 不能同时指定 before 和 after。")
        name = operation.get("name")
        if name == "":
            name = None
        if name is not None and (
            not isinstance(name, str)
            or "." in name
            or len(name) > 256
        ):
            raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path}.name 无效。")
        layer = _variant_layer(operation, path)
        return {
            "action": action,
            "parent": _variant_path(operation.get("parent"), f"{path}.parent"),
            "name": name,
            "before": _variant_path(before, f"{path}.before") if before is not None else None,
            "after": _variant_path(after, f"{path}.after") if after is not None else None,
            "initialization_seed": _layer_seed(operation, path),
            **_layer_rng_devices(operation),
            **layer,
        }
    if action == "replace":
        return {
            "action": action,
            "path": _variant_child_path(operation.get("path"), f"{path}.path"),
            "initialization_seed": _layer_seed(operation, path),
            **_layer_rng_devices(operation),
            **_variant_layer(operation, path),
        }
    if action == "set":
        params = _parameters(operation.get("params"), f"{path}.params")
        if not params:
            raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path}.params 不能为空。")
        return {"action": action, "path": _variant_child_path(operation.get("path"), f"{path}.path"), "params": params}
    if action == "remove":
        strategy = operation.get("strategy", "remove")
        if strategy not in {"remove", "bypass"}:
            raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path}.strategy 必须为 remove 或 bypass。")
        return {"action": action, "path": _variant_child_path(operation.get("path"), f"{path}.path"), "strategy": strategy}
    raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path} 含不支持的 Layer 操作：{action!r}。")


def _layer_rng_devices(operation: dict) -> dict:
    if "rng_devices" not in operation:
        return {}
    validate_rng_devices(operation["rng_devices"])
    return {"rng_devices": list(operation["rng_devices"])}


def _layer_seed(operation: dict, path: str) -> int:
    seed = operation.get("initialization_seed")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", f"{path}.initialization_seed 必须是无符号 64 位整数。")
    return seed


def _variant(value: Any) -> dict:
    if not isinstance(value, dict) or not {"seed", "edits"} <= set(value) or set(value) - {"seed", "edits", "device_policy", "base_rng_devices"}:
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", "variant 需要 seed、活动 edits 和可选设备策略。")
    seed, edits = value["seed"], value["edits"]
    device_policy = value.get("device_policy", "legacy_cpu")
    if not isinstance(device_policy, str) or device_policy not in {"legacy_cpu", "factory"}:
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", "variant.device_policy 无效。")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", "variant.seed 必须是无符号 64 位整数。")
    if not isinstance(edits, list) or len(edits) > 128:
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", "variant.edits 最多包含 128 项活动历史。")
    normalized = {
        "seed": seed,
        "edits": [_variant_operation(item, f"variant.edits[{index}]") for index, item in enumerate(edits)],
    }
    if "device_policy" in value:
        normalized["device_policy"] = device_policy
    if "base_rng_devices" in value:
        validate_rng_devices(value["base_rng_devices"])
        normalized["base_rng_devices"] = list(value["base_rng_devices"])
    if len(json.dumps(normalized, ensure_ascii=False, allow_nan=False).encode("utf-8")) > _MAX_RECIPE_BYTES:
        raise UserError("ASSEMBLY_VARIANT_RECIPE_INVALID", "Layer 变体编辑配方超过 1 MiB。")
    return normalized


_VARIANT_HELPERS = '''\
def _module_at(model, path):
    parts = path.split(".")
    if parts and parts[0] == "model":
        parts = parts[1:]
    current = model
    for part in parts:
        children = getattr(current, "_modules", {})
        if part not in children or children[part] is None:
            raise ValueError(f"Layer path not found: {path}")
        current = children[part]
    return current


def _parent_slot(model, path):
    parent_path, _, name = path.rpartition(".")
    if not parent_path or parent_path == "model":
        parent_path = "model"
    parent = _module_at(model, parent_path)
    if name not in parent._modules or parent._modules[name] is None:
        raise ValueError(f"Layer path not found: {path}")
    return parent, name


def _replace_module(model, path, replacement):
    old = _module_at(model, path)
    if old is model:
        raise ValueError("The root model cannot be replaced by a Layer edit")
    stack = [model]
    visited = set()
    replaced = 0
    while stack:
        parent = stack.pop()
        if id(parent) in visited:
            continue
        visited.add(id(parent))
        for name, child in list(parent._modules.items()):
            if child is old:
                parent._modules[name] = replacement
                replaced += 1
            elif child is not None:
                stack.append(child)
    if replaced == 0:
        raise ValueError(f"Layer path is not registered: {path}")
    return old


def _new_layer(layer_type, component, params, seed, preserve_factory_device=False, rng_devices=()):
    factory = _resolve(component) if component else getattr(torch.nn, layer_type)
    if component is None and (not isinstance(factory, type) or not issubclass(factory, torch.nn.Module)):
        raise TypeError(f"torch.nn.{layer_type} is not a Module class")
    backend = rng_devices[0].split(":", 1)[0] if rng_devices else None
    indexes = [int(device.split(":", 1)[1]) for device in rng_devices]
    fork = (torch.random.fork_rng(devices=indexes, device_type=backend) if backend
            else torch.random.fork_rng(devices=[]))
    with fork:
        torch.random.default_generator.manual_seed(seed)
        if backend:
            generators = getattr(getattr(torch, backend), "default_generators")
            for index in indexes:
                generators[index].manual_seed(seed)
        layer = factory(**params)
    if not isinstance(layer, torch.nn.Module):
        raise TypeError("Layer factory did not return torch.nn.Module")
    return layer if preserve_factory_device else layer.cpu()


def _editable_parameters(module):
    nn = torch.nn
    if type(module) is nn.Linear:
        return {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
    if type(module) is nn.Dropout:
        return {"p": module.p, "inplace": module.inplace}
    if type(module) is nn.ReLU:
        return {"inplace": module.inplace}
    if type(module) is nn.GELU:
        return {"approximate": module.approximate}
    if type(module) is nn.LayerNorm:
        return {
            "normalized_shape": list(module.normalized_shape),
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine,
            "bias": module.bias is not None,
        }
    if type(module) is nn.Conv2d:
        padding = (
            module.padding
            if isinstance(module.padding, str)
            else list(module.padding)
        )
        return {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size),
            "stride": list(module.stride),
            "padding": padding,
            "dilation": list(module.dilation),
            "groups": module.groups,
            "bias": module.bias is not None,
            "padding_mode": module.padding_mode,
        }
    if type(module) is nn.Identity:
        return {}
    raise TypeError(f"Layer does not have a built-in parameter editor: {type(module).__name__}")


def _preserve_state(old, replacement, preserve_matching=True):
    state = next(old.parameters(), None)
    if state is None:
        state = next(old.buffers(), None)
    if state is not None:
        if state.is_floating_point() or state.is_complex():
            replacement.to(device=state.device, dtype=state.dtype)
        else:
            replacement.to(device=state.device)
    replacement.train(old.training)
    current = replacement.state_dict()
    previous = old.state_dict()
    for name, tensor in current.items():
        if preserve_matching and name in previous and tensor.shape == previous[name].shape:
            tensor.copy_(previous[name])
    replacement.load_state_dict(current)


def _storage_key(tensor):
    try:
        if str(tensor.layout) != "torch.strided" or tensor.device.type == "meta":
            return None
        storage = tensor.untyped_storage()
        return (str(tensor.device), storage._cdata) if storage.nbytes() else None
    except (AttributeError, RuntimeError, TypeError, NotImplementedError):
        return None


def _preserve_external_sharing(model, old, replacement, compatible):
    aliases = _registered_paths(model, old)
    outside_objects, outside_storages = set(), {}
    for accessor in ("named_parameters", "named_buffers"):
        for path, tensor in getattr(model, accessor)(remove_duplicate=False):
            if any(("model." + path).startswith(alias + ".") for alias in aliases):
                continue
            outside_objects.add(id(tensor))
            key = _storage_key(tensor)
            if key is not None:
                outside_storages.setdefault(key, set()).add(id(tensor))
    shared = []
    for kind, accessor in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
        for name, tensor in getattr(old, accessor)(remove_duplicate=False):
            key = _storage_key(tensor)
            storage_shared = key is not None and any(other != id(tensor) for other in outside_storages.get(key, ()))
            if id(tensor) not in outside_objects and not storage_shared:
                continue
            parent_path, _, local_name = name.rpartition(".")
            try:
                parent = replacement.get_submodule(parent_path) if parent_path else replacement
            except AttributeError:
                parent = None
            registry = None if parent is None else (parent._parameters if kind == "parameter" else parent._buffers)
            target = registry.get(local_name) if registry is not None else None
            if not compatible or target is None or target.shape != tensor.shape or target.dtype != tensor.dtype:
                raise ValueError(f"Layer edit would break externally shared {kind}: {name}")
            shared.append((registry, local_name, tensor))
    for registry, name, tensor in shared:
        registry[name] = tensor


def _insert_layer(model, operation, preserve_factory_device=False):
    parent = _module_at(model, operation["parent"])
    if isinstance(parent, torch.nn.ModuleList):
        kind = "list"
    elif isinstance(parent, torch.nn.ModuleDict):
        kind = "dict"
    elif isinstance(parent, torch.nn.Sequential):
        kind = "sequential"
    else:
        raise TypeError(
            "Layer insertion requires a native module container, "
            f"got {type(parent).__name__}"
        )
    name = operation.get("name")
    before, after = operation.get("before"), operation.get("after")
    anchor = (before or after).rsplit(".", 1)[-1] if before or after else None
    layer = _new_layer(
        operation.get("layer_type"),
        operation.get("component"),
        operation["params"],
        operation["initialization_seed"],
        preserve_factory_device,
        operation.get("rng_devices", ()),
    )
    items = list(parent._modules.items())
    if kind == "list":
        if name not in (None, ""):
            raise ValueError("ModuleList insertion does not accept a name")
        if anchor is None:
            index = len(items)
        else:
            index = next((i for i, (key, _) in enumerate(items) if key == anchor), None)
            if index is None:
                raise ValueError(f"ModuleList anchor not found: {anchor}")
            if after:
                index += 1
        items.insert(index, ("", layer))
        items = [(str(i), child) for i, (_, child) in enumerate(items)]
    else:
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError(f"Invalid child name: {name!r}")
        if name in parent._modules or (kind == "sequential" and hasattr(parent, name)):
            raise ValueError(f"Child name already exists: {name}")
        if anchor is not None and anchor not in parent._modules:
            raise ValueError(f"Container anchor not found: {anchor}")
        rebuilt = []
        for key, child in items:
            if before and key == anchor:
                rebuilt.append((name, layer))
            rebuilt.append((key, child))
            if after and key == anchor:
                rebuilt.append((name, layer))
        if anchor is None:
            rebuilt.append((name, layer))
        items = rebuilt
    parent._modules.clear()
    parent._modules.update(items)


def _remove_layer(model, path, strategy):
    old = _module_at(model, path)
    if old is model:
        raise ValueError("The root model cannot be removed")
    if strategy == "bypass":
        replacement = torch.nn.Identity()
        _preserve_state(old, replacement)
        _preserve_external_sharing(model, old, replacement, False)
        _replace_module(model, path, replacement)
        return
    parent, name = _parent_slot(model, path)
    if not isinstance(parent, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)):
        raise TypeError(
            "Layer removal requires a native module container, "
            f"got {type(parent).__name__}"
        )
    aliases = _registered_paths(model, old)
    if len(aliases) > 1:
        raise ValueError(f"Cannot remove a shared Layer with aliases: {aliases}")
    del parent._modules[name]
    if isinstance(parent, torch.nn.ModuleList):
        remaining = list(parent._modules.values())
        parent._modules.clear()
        parent._modules.update((str(i), child) for i, child in enumerate(remaining))


def _registered_paths(model, target):
    paths = []
    stack = [(model, "model", frozenset())]
    while stack:
        parent, prefix, ancestors = stack.pop()
        if id(parent) in ancestors:
            continue
        next_ancestors = ancestors | {id(parent)}
        for name, child in parent._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{name}"
            if child is target:
                paths.append(path)
            stack.append((child, path, next_ancestors))
    return sorted(paths)


def _apply_edit(model, operation, preserve_factory_device=False):
    action = operation["action"]
    if action == "transaction":
        for child in operation["operations"]:
            _apply_edit(model, child, preserve_factory_device)
    elif action == "add":
        _insert_layer(model, operation, preserve_factory_device)
    elif action == "set":
        old = _module_at(model, operation["path"])
        params = _editable_parameters(old)
        if set(operation["params"]) - set(params):
            unsupported = sorted(set(operation["params"]) - set(params))
            raise ValueError(
                f"Unsupported parameter(s) for {type(old).__name__}: {unsupported}"
            )
        params.update(operation["params"])
        replacement = type(old)(**params)
        _preserve_state(old, replacement)
        _preserve_external_sharing(model, old, replacement, True)
        _replace_module(model, operation["path"], replacement)
    elif action == "replace":
        old = _module_at(model, operation["path"])
        replacement = _new_layer(
            operation.get("layer_type"),
            operation.get("component"),
            operation["params"],
            operation["initialization_seed"],
            preserve_factory_device,
            operation.get("rng_devices", ()),
        )
        _preserve_state(old, replacement, preserve_matching=False)
        _preserve_external_sharing(model, old, replacement, False)
        _replace_module(model, operation["path"], replacement)
    elif action == "remove":
        _remove_layer(model, operation["path"], operation["strategy"])
    elif action == "sharing":
        if operation["mode"] == "clone-selected":
            import copy
            parent, name = _parent_slot(model, operation["path"])
            parent._modules[name] = copy.deepcopy(parent._modules[name])
        else:
            source = _module_at(model, operation["source"])
            for target_path in operation["targets"]:
                parent, name = _parent_slot(model, target_path)
                parent._modules[name] = source
    else:
        raise ValueError(f"Unsupported Layer edit: {action}")


def _apply_variant(model, edits, preserve_factory_device=False):
    for operation in edits:
        _apply_edit(model, operation, preserve_factory_device)
'''


def validate_recipe(recipe: Any) -> dict:
    """Validate and normalize an explicit, dependency-light assembly recipe."""
    if not isinstance(recipe, dict) or set(recipe) - {
        "schema_version", "model", "model_injections", "initialization_seed", "description", "variant"
    }:
        raise UserError("ASSEMBLY_RECIPE_INVALID", "装配配方不能包含未知字段。")
    schema_version = recipe.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise UserError("ASSEMBLY_RECIPE_INVALID", "assembly recipe schema_version 必须为 1 或 2。")
    if (schema_version == 2) != ("variant" in recipe):
        raise UserError("ASSEMBLY_RECIPE_INVALID", "包含 variant 时必须使用 schema_version=2；v1 不包含 variant。")
    model = recipe.get("model")
    if not isinstance(model, dict) or set(model) - {"factory", "params"}:
        raise UserError("ASSEMBLY_RECIPE_INVALID", "model 必须包含 factory 和可选 params。")
    normalized_model = {
        "factory": _entrypoint(model.get("factory"), "model.factory"),
        "params": _parameters(model.get("params", {}), "model.params"),
    }
    injections = recipe.get("model_injections", {})
    if not isinstance(injections, dict) or len(injections) > 32:
        raise UserError("ASSEMBLY_RECIPE_INVALID", "model_injections 必须是最多 32 项的对象。")
    normalized_injections = {}
    for slot, specification in injections.items():
        if not isinstance(slot, str) or not slot.isidentifier():
            raise UserError("ASSEMBLY_RECIPE_INVALID", f"无效的模型接缝名：{slot!r}。")
        if slot in normalized_model["params"]:
            raise UserError(
                "ASSEMBLY_RECIPE_INVALID",
                f"model.params 与 model_injections 重复使用关键字 {slot}。",
            )
        if isinstance(specification, str):
            factory, params, component_ref = specification, {}, None
        elif isinstance(specification, dict) and not set(specification) - {
            "factory", "params", "component_ref"
        }:
            factory = specification.get("factory")
            params = specification.get("params", {})
            component_ref = specification.get("component_ref")
        else:
            raise UserError("ASSEMBLY_RECIPE_INVALID", f"接缝 {slot} 的声明格式无效。")
        normalized = {
            "factory": _entrypoint(factory, f"model_injections.{slot}.factory"),
            "params": _parameters(params, f"model_injections.{slot}.params"),
        }
        if component_ref is not None:
            if (
                not isinstance(component_ref, dict)
                or not isinstance(component_ref.get("id"), str)
                or not component_ref.get("id")
                or len(component_ref.get("id", "")) > 256
                or any(ord(char) < 32 for char in component_ref.get("id", ""))
                or not isinstance(component_ref.get("version"), str)
                or not component_ref.get("version")
                or len(component_ref.get("version", "")) > 128
                or any(ord(char) < 32 for char in component_ref.get("version", ""))
                or not isinstance(component_ref.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", component_ref.get("sha256", ""))
            ):
                raise UserError("ASSEMBLY_RECIPE_INVALID", f"接缝 {slot} 的组件锁格式无效。")
            normalized["component_ref"] = {
                "id": component_ref["id"],
                "version": component_ref["version"],
                "sha256": component_ref["sha256"],
            }
        normalized_injections[slot] = normalized
    seed = recipe.get("initialization_seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise UserError("ASSEMBLY_RECIPE_INVALID", "initialization_seed 必须是无符号 64 位整数。")
    description = recipe.get("description", "")
    if not isinstance(description, str) or len(description) > 512:
        raise UserError("ASSEMBLY_RECIPE_INVALID", "description 必须是最多 512 字符的文本。")
    normalized = {
        "schema_version": schema_version,
        "model": normalized_model,
        "model_injections": normalized_injections,
        "description": description,
    }
    if seed is not None:
        normalized["initialization_seed"] = seed
    if "variant" in recipe:
        normalized["variant"] = _variant(recipe["variant"])
    encoded = json.dumps(normalized, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_RECIPE_BYTES:
        raise UserError("ASSEMBLY_RECIPE_INVALID", "装配配方超过 1 MiB。")
    return normalized


def generate_code(recipe: Any) -> str:
    """Return readable Python for a declared assembly and supported edit recipe.

    The generated module is a new artifact; callers must choose a separate
    destination and must not overwrite project-owned source files.
    """
    normalized = validate_recipe(recipe)
    variant = normalized.get("variant")
    lines = [
        "\"\"\"Generated Model-Maven assembly adapter.\"\"\"",
        f"# Generated from an explicit Model-Maven assembly recipe (schema v{normalized['schema_version']}).",
        "# This adapter does not vendor user modules, extension packages, or runtime dependencies.",
        "# Make the project's module directories available to the selected Python environment.",
        "# Use the working directory expected by user factories and relative data paths.",
        "from importlib import import_module",
        "",
        "",
        "def _resolve(entrypoint):",
        "    module_name, attribute = entrypoint.split(\":\", 1)",
        "    value = import_module(module_name)",
        "    for part in attribute.split(\".\"):",
        "        value = getattr(value, part)",
        "    return value",
    ]
    if variant:
        lines.extend(["", "import random", "import torch", "", ""])
        lines.extend(_VARIANT_HELPERS.strip("\n").splitlines())
    elif normalized.get("initialization_seed") is not None:
        lines.extend(["", "import random", "import torch"])
    lines.extend(["", "", "def build_model():"])
    seed = variant["seed"] if variant else normalized.get("initialization_seed")
    if variant:
        base_rng_devices = variant.get("base_rng_devices")
        backend, indices = validate_rng_devices(base_rng_devices or [])
        fork = (f"torch.random.fork_rng(devices={indices!r}, device_type={backend!r})"
                if indices else "torch.random.fork_rng(devices=[])")
        base_scope_comment = ("    # Base construction seeds CPU and the explicitly recorded accelerator RNGs."
                              if indices else "    # Base construction seeds Python and CPU only; accelerator RNGs are not seeded here.")
        lines.extend([
            base_scope_comment,
            "    # Match any device-specific initialization deliberately in your project code.",
            "    python_rng_state = random.getstate()",
            f"    random.seed({seed})",
            "    try:",
            f"        with {fork}:",
            f"            torch.random.default_generator.manual_seed({seed})",
            *[f"            getattr(torch, {backend!r}).default_generators[{index}].manual_seed({seed})" for index in indices],
            "            model = _build_model()",
            *([] if variant.get("device_policy", "legacy_cpu") == "factory" else [
                "            # Match the persisted variant's CPU reconstruction policy.",
                "            model.cpu()",
            ]),
            f"            _apply_variant(model, {variant['edits']!r}, {variant.get('device_policy') == 'factory'!r})",
            "    finally:",
            "        random.setstate(python_rng_state)",
            "    return model",
        ])
        lines.extend(["", "", "def _build_model():"])
    elif seed is not None:
        lines.extend([
            "    # Base construction seeds Python and CPU only; accelerator RNGs are not seeded here.",
            "    # Match any device-specific initialization deliberately in your project code.",
            "    python_rng_state = random.getstate()",
            f"    random.seed({seed})",
            "    try:",
            "        with torch.random.fork_rng(devices=[]):",
            f"            torch.random.default_generator.manual_seed({seed})",
            "            model = _build_model()",
            "    finally:",
            "        random.setstate(python_rng_state)",
            "    return model",
            "",
            "",
            "def _build_model():",
        ])
    else:
        lines.extend(["    return _build_model()", "", "", "def _build_model():"])
    params = repr(normalized["model"]["params"])
    if normalized["description"]:
        safe_description = normalized["description"].replace("\r", " ").replace("\n", " ")
        lines.append(f"    # Description: {json.dumps(safe_description, ensure_ascii=True)}")
    injections = normalized["model_injections"]
    for index, (slot, spec) in enumerate(injections.items()):
        lines.append(f"    # Declared root-model seam: {slot}")
        reference = spec.get("component_ref")
        if reference:
            lines.append(
                "    # Component lock: "
                f"{json.dumps(reference['id'], ensure_ascii=True)}@"
                f"{json.dumps(reference['version'], ensure_ascii=True)} "
                f"sha256:{reference['sha256']}"
            )
        lines.append(
            f"    _component_{index} = _resolve({spec['factory']!r})(**{repr(spec['params'])})"
        )
    kwargs = [f"**{params}"] if params else []
    if injections:
        entries = ", ".join(f"{slot!r}: _component_{index}" for index, slot in enumerate(injections))
        lines.append(f"    _components = {{{entries}}}")
        kwargs.append("**_components")
    call = f"_resolve({normalized['model']['factory']!r})({', '.join(kwargs)})"
    lines.append(f"    return {call}")
    lines.extend(["", "", "if __name__ == \"__main__\":", "    model = build_model()", "    print(type(model).__qualname__)", ""])
    return "\n".join(lines)


def write_code(recipe: Any, destination: str | Path) -> Path:
    """Create a new generated file without replacing any existing file."""
    path = Path(destination)
    if not path.name or path.exists():
        raise UserError(
            "ASSEMBLY_OUTPUT_EXISTS",
            f"装配代码目标已存在：{path}。生成器不会覆盖现有源码或产物。",
            "选择一个新的 generated/ 路径，或由用户显式整理旧产物后再生成。",
        )
    code = generate_code(recipe).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError as error:
        raise UserError(
            "ASSEMBLY_OUTPUT_EXISTS",
            f"装配代码目标已存在：{path}。生成器不会覆盖现有源码或产物。",
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(code)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path
