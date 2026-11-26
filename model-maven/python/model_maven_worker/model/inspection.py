"""Inspectable Layer parameters and bounded model structure reports."""

import inspect
import json
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
    # For other public, stateless PyTorch layers, offer only constructor
    # arguments that are directly represented as JSON-compatible instance
    # attributes. Do not infer reconstruction for custom subclasses, hidden
    # constructor state, containers, or layers with parameters/buffers.
    cls = type(module)
    if (not cls.__name__.startswith("_") and getattr(nn, cls.__name__, None) is cls
            and not module._modules and next(module.parameters(), None) is None
            and next(module.buffers(), None) is None):
        try:
            signature = inspect.signature(cls.__init__)
            values = {}
            for name, param in signature.parameters.items():
                if name == "self":
                    continue
                if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    return None
                if name not in vars(module):
                    return None
                values[name] = vars(module)[name]
            if not values:
                return None
            return json.loads(json.dumps(values, allow_nan=False))
        except (OSError, TypeError, ValueError, OverflowError, RecursionError):
            return None
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


def _source_edit_details(module, torch_module):
    """Share safe state/signature checks for normal PyTorch traversal."""
    base = torch_module.nn.Module
    names = ("parameters", "buffers", "named_parameters", "named_buffers", "named_modules", "_named_members")
    if (not isinstance(module, base)
            or not isinstance(getattr(module, "_modules", None), dict)
            or bool(module._modules)
            or any(getattr(type(module), name, None) is not getattr(base, name, None) for name in names)
            or any(name in vars(module) for name in names)):
        # User overrides can make individual traversals observable or dynamic.
        return source_editable_fields(module, torch_module), source_set_block_reason(module, torch_module)
    known = editable_params(module, torch_module)
    if known is not None:
        return set(known), None
    if has_module_state(module):
        return set(), "该自定义 Layer 含参数或 Buffer；源码改参需要显式状态迁移处理器。"
    fields = source_constructor_fields(module)
    return fields, None if fields else "构造器没有可安全编辑的显式关键字参数。"


def _state_placement(model):
    """Summarize registered tensors without reading values or moving devices."""
    counts = {}
    observed = 0
    partial = False
    try:
        for kind, values in (("parameter", model.parameters()), ("buffer", model.buffers())):
            for value in values:
                if observed >= 4096:
                    partial = True
                    break
                key = (kind, str(value.device), str(value.dtype))
                if key not in counts and len(counts) >= 32:
                    partial = True
                    break
                counts[key] = counts.get(key, 0) + 1
                observed += 1
            if partial:
                break
    except Exception:
        partial = True
    return {
        "status": "partial" if partial else "observed",
        "observed_tensors": observed,
        "groups": [{"kind": kind, "device": device, "dtype": dtype, "tensors": count}
                   for (kind, device, dtype), count in sorted(counts.items())],
        "coverage": "registered parameters and buffers only; shared objects deduplicated within each kind; up to 4096 tensors and 32 groups",
    }


def _parameter_count(parameters):
    """Do not report a partial count as a total for lazy or custom state."""
    try:
        total = 0
        for parameter in parameters:
            size = parameter.numel()
            if type(size) is not int or size < 0:
                return None
            total += size
        return total
    except Exception:
        return None


def model_state_summary(model):
    return {
        "placement": _state_placement(model),
        "parameters": _parameter_count(model.parameters()),
        "trainable_parameters": _parameter_count(p for p in model.parameters() if p.requires_grad),
    }


def _tree_parameter_counts(modules, aliases, torch_module):
    """Count a plain unshared module tree once; otherwise retain PyTorch traversal."""
    if any(len(paths) != 1 for paths in aliases.values()):
        return None
    module_base = torch_module.nn.Module
    counts, seen, state_count = {}, set(), 0
    try:
        for key, module in reversed(list(modules.items())):
            if (type(module).parameters is not module_base.parameters
                    or type(module).named_parameters is not module_base.named_parameters
                    or type(module).named_modules is not module_base.named_modules
                    or any(name in vars(module) for name in ("parameters", "named_parameters", "named_modules"))
                    or not isinstance(module._parameters, dict) or not isinstance(module._modules, dict)):
                return None
            own = 0
            for parameter in module._parameters.values():
                if parameter is None:
                    continue
                if type(parameter) is not torch_module.nn.Parameter:
                    return None
                state_count += 1
                if state_count > 200000 or id(parameter) in seen:
                    return None
                seen.add(id(parameter))
                try:
                    size = parameter.numel()
                    if type(size) is not int or size < 0:
                        own = None
                    elif own is not None:
                        own += size
                except Exception:
                    own = None
            total = own
            for child in module._modules.values():
                if child is None:
                    continue
                child_count = counts.get(id(child))
                if child_count is None:
                    return None
                child_total = child_count[1]
                total = total + child_total if total is not None and child_total is not None else None
            counts[key] = (own, total)
    except Exception:
        return None
    return counts


def inspect_model(model, path=None, layer_type=None, search=None, *, torch_module, source_locations, source_site_counts, layer_container_kind_fn):
    modules, aliases = module_entries(model)
    selected = normalize_path(path)
    search_term = search.strip().casefold() if isinstance(search, str) else ""
    resolve_module(model, selected)
    fast_counts = (_tree_parameter_counts(modules, aliases, torch_module)
                   if selected == "model" and not layer_type and not search_term else None)
    rows = []
    definitions = {}
    source_files = {}
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
        source_fields, source_set_reason = (_source_edit_details(module, torch_module)
                                            if site is not None else (set(), None))
        site_key = (
            type(module).__module__, type(module).__qualname__,
            site.get("source_path"), site.get("line"),
        ) if site is not None else None
        module_type = type(module)
        if module_type not in source_files:
            try:
                source_files[module_type] = inspect.getsourcefile(module_type)
            except (OSError, TypeError):
                source_files[module_type] = None
        filename = source_files[module_type]
        if module_type not in definitions:
            try:
                _, first_line = inspect.getsourcelines(module_type)
                definitions[module_type] = (
                    {"file": str(Path(filename).resolve()), "line": first_line}
                    if filename
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
            "own_parameters": fast_counts[key][0] if fast_counts is not None else _parameter_count(module.parameters(recurse=False)),
            "parameters": fast_counts[key][1] if fast_counts is not None else _parameter_count(module.parameters()),
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
    state = model_state_summary(model)
    return {"modules": rows, "model_state_placement": state["placement"], "parameters": state["parameters"],
            "trainable_parameters": state["trainable_parameters"]}


