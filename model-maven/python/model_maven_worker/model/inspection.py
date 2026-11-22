"""Inspectable Layer parameters and bounded model structure reports."""

import inspect
from pathlib import Path

from ..protocol import UserError
from .structure import module_entries, normalize_path, resolve_module


def editable_params(module, torch_module):
    nn = torch_module.nn
    if type(module) is nn.Linear:
        return {"in_features": module.in_features, "out_features": module.out_features, "bias": module.bias is not None}
    if type(module) is nn.Dropout:
        return {"p": module.p, "inplace": module.inplace}
    if type(module) is nn.ReLU:
        return {"inplace": module.inplace}
    if type(module) is nn.GELU:
        return {"approximate": module.approximate}
    if type(module) is nn.LayerNorm:
        return {"normalized_shape": list(module.normalized_shape), "eps": module.eps,
                "elementwise_affine": module.elementwise_affine, "bias": module.bias is not None}
    if type(module) is nn.Conv2d:
        return {"in_channels": module.in_channels, "out_channels": module.out_channels,
                "kernel_size": list(module.kernel_size), "stride": list(module.stride),
                "padding": list(module.padding) if not isinstance(module.padding, str) else module.padding,
                "dilation": list(module.dilation), "groups": module.groups,
                "bias": module.bias is not None, "padding_mode": module.padding_mode}
    if type(module) is nn.Identity:
        return {}
    return None


def source_constructor_fields(module):
    try:
        signature = inspect.signature(type(module).__init__)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def has_module_state(module):
    return next(module.parameters(), None) is not None or next(module.buffers(), None) is not None


def source_editable_fields(module, torch_module):
    """Return constructor fields safe for generic source edits without state migration."""
    known = editable_params(module, torch_module)
    if known is not None:
        return set(known)
    if not isinstance(module, torch_module.nn.Module):
        return set()
    if has_module_state(module):
        return set()
    return source_constructor_fields(module)


def source_set_block_reason(module, torch_module):
    if editable_params(module, torch_module) is not None or not isinstance(module, torch_module.nn.Module):
        return None
    if has_module_state(module):
        return "该自定义 Layer 含参数或 Buffer；源码改参需要显式状态迁移处理器。"
    if not source_constructor_fields(module):
        return "构造器没有可安全编辑的显式关键字参数。"
    return None


def inspect_model(model, path=None, layer_type=None, search=None, *, torch_module, source_locations, source_site_counts, layer_container_kind_fn):
    modules, aliases = module_entries(model)
    selected = normalize_path(path)
    search_term = search.strip().casefold() if isinstance(search, str) else ""
    resolve_module(model, selected)
    rows = []
    definitions = {}
    for index, (key, module) in enumerate(modules.items()):
        relevant = [p for p in aliases[key] if p == selected or p.startswith(selected + ".")]
        if not relevant or (layer_type and layer_type.lower() not in type(module).__name__.lower()):
            continue
        if search_term and not any(
            search_term in value.casefold()
            for value in [*relevant, type(module).__name__]
        ):
            continue
        params = editable_params(module, torch_module)
        site = source_locations.get(key)
        source_fields = source_editable_fields(module, torch_module) if site is not None else set()
        source_set_reason = source_set_block_reason(module, torch_module) if site is not None else None
        site_key = (
            type(module).__module__, type(module).__qualname__,
            site.get("source_path"), site.get("line"),
        ) if site is not None else None
        try:
            filename = inspect.getsourcefile(type(module))
        except (OSError, TypeError):
            filename = None
        module_type = type(module)
        if module_type not in definitions:
            try:
                _, first_line = inspect.getsourcelines(module_type)
                source_file = inspect.getsourcefile(module_type)
                definitions[module_type] = (
                    {"file": str(Path(source_file).resolve()), "line": first_line}
                    if source_file
                    else None
                )
            except (OSError, TypeError, ValueError):
                definitions[module_type] = None
        rows.append({
            "path": relevant[0], "aliases": aliases[key], "type": type(module).__name__,
            # Only the built-in Sequential guarantees this registration order
            # is its forward order. Subclasses may override forward.
            **({"sequential_children": list(module._modules)}
               if type(module) is torch_module.nn.Sequential
               else {"sequential_children": None} if type(module).__name__ == "Sequential"
               else {}),
            "instance_id": f"module-{index + 1}",
            "own_parameters": sum(p.numel() for p in module.parameters(recurse=False)),
            "parameters": sum(p.numel() for p in module.parameters()),
            "editable_params": params, "source": filename,
            "source_editable_fields": sorted(source_fields),
            "source_set_block_reason": source_set_reason,
            "can_source_set": bool(
                module is not model
                and source_fields
                and source_site_counts.get(site_key, 0) == 1
            ),
            "source_definition": definitions[module_type],
            "construction_site": site,
            "can_set": params is not None,
            "can_insert": layer_container_kind_fn(module) is not None,
            "can_source_insert": type(module) in (
                torch_module.nn.Sequential,
                torch_module.nn.ModuleList,
                torch_module.nn.ModuleDict,
            ),
        })
    return {"modules": rows, "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


