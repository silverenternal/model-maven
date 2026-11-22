"""Stable paths and identities for registered model modules."""

from ..protocol import UserError


def module_entries(model):
    aliases = {}
    modules = {}
    try:
        for index, (name, module) in enumerate(model.named_modules(remove_duplicate=False)):
            if index >= 4096:
                raise UserError("MODEL_TOO_LARGE", "模块别名超过当前预览上限 4096。")
            path = "model" + ("." + name if name else "")
            key = id(module)
            aliases.setdefault(key, []).append(path)
            modules[key] = module
    except RecursionError as error:
        raise UserError("MODULE_CYCLE_UNSUPPORTED", "当前结构检查不支持循环注册的子模块。") from error
    return modules, aliases


def normalize_path(path):
    if path in (None, "", "model"):
        return "model"
    return path if path.startswith("model.") else "model." + path


def resolve_module(model, path):
    path = normalize_path(path)
    current = model
    for part in path.split(".")[1:]:
        if part not in current._modules:
            raise UserError("LAYER_NOT_FOUND", f"找不到 Layer：{path}", "先执行 layer list 或 model inspect 查看实际路径。")
        current = current._modules[part]
    return current




def registered_storage_key(tensor):
    """Identity of inspectable nonempty strided storage, local to this process."""
    try:
        if str(tensor.layout) != "torch.strided" or tensor.device.type == "meta":
            return None
        storage = tensor.untyped_storage()
        if storage.nbytes() == 0:
            return None
        return (str(tensor.device), storage._cdata)
    except (AttributeError, RuntimeError, TypeError, NotImplementedError):
        return None


def shared_state_groups(model):
    """Bounded registered object and strided-storage alias evidence."""
    groups, objects, storages, unsupported = [], {}, {}, []
    seen_paths = 0
    for kind, accessor in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
        for name, tensor in getattr(model, accessor)(remove_duplicate=False):
            seen_paths += 1
            if seen_paths > 200_000:
                raise UserError("SHARED_STATE_EVIDENCE_TOO_LARGE", "模型注册状态超过 200000 项，无法完整核对共享状态。")
            path = "model." + name
            objects.setdefault((kind, id(tensor)), []).append(path)
            key = registered_storage_key(tensor)
            if key is None:
                unsupported.append({"kind": kind, "path": path})
                continue
            try:
                view = {"kind": kind, "path": path, "shape": list(tensor.shape),
                        "stride": list(tensor.stride()), "offset": tensor.storage_offset(),
                        "dtype": str(tensor.dtype)}
            except (AttributeError, RuntimeError, TypeError, NotImplementedError):
                unsupported.append({"kind": kind, "path": path})
                continue
            storages.setdefault(key, []).append(view)
    for (kind, _), paths in objects.items():
        if len(paths) > 1:
            groups.append({"kind": kind, "paths": sorted(paths)})
    for views in storages.values():
        if len(views) > 1:
            groups.append({"kind": "storage", "paths": sorted(view["path"] for view in views),
                           "views": sorted(views, key=lambda view: (view["kind"], view["path"]))})
    # An explicit unknown entry prevents an unsupported tensor from being silently
    # presented as a checked, unshared storage.
    groups.extend({"kind": "storage_unavailable", "paths": [item["path"]], "state_kind": item["kind"]} for item in unsupported)
    groups.sort(key=lambda item: (item["kind"], item["paths"], item.get("state_kind", "")))
    if sum(len(item["paths"]) for item in groups) > 16384:
        raise UserError("SHARED_STATE_EVIDENCE_TOO_LARGE", "共享或无法检查的状态路径超过 16384 项，无法完整核对。")
    return groups
