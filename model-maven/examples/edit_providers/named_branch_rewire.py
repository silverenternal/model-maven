"""Example API v2 Provider for connecting named module branches."""

MODEL_MAVEN_LAYER_EDIT_API = 2


def _resolve(model, path):
    normalized = path if path == "model" or path.startswith("model.") else f"model.{path}"
    if normalized == "model":
        return model
    return model.get_submodule(normalized.removeprefix("model."))


def _parent(model, path):
    normalized = path if path.startswith("model.") else f"model.{path}"
    parent_path, _, name = normalized.rpartition(".")
    parent = _resolve(model, parent_path)
    if name not in parent._modules:
        raise ValueError(f"{normalized} is not a registered child module")
    return normalized, parent, name


class NamedBranchRewire:
    def __init__(self):
        self._previous = None

    def describe(self, model, operation):
        return {
            "name": "named-branch-rewire",
            "targeting": "registered module paths",
            "dynamic_invocation_indexes": False,
        }

    def can_edit(self, model, operation):
        payload = operation.get("payload", {})
        source, target = payload.get("source"), payload.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            return {"state": "unsupported", "reason": "payload needs named source and target paths"}
        try:
            _resolve(model, source)
            _parent(model, target)
        except (AttributeError, KeyError, ValueError) as error:
            return {"state": "unsupported", "reason": str(error)}
        return {"state": "supported", "reason": "both named module paths resolve"}

    def plan(self, model, operation):
        payload = operation["payload"]
        source, _, _ = _parent(model, payload["source"])
        target, _, _ = _parent(model, payload["target"])
        return {"source": source, "target": target}

    def candidate(self, model, plan):
        _, source_parent, source_name = _parent(model, plan["source"])
        target, target_parent, target_name = _parent(model, plan["target"])
        self._previous = (target_parent, target_name, target_parent._modules[target_name])
        target_parent._modules[target_name] = source_parent._modules[source_name]
        return {
            "source": plan["source"],
            "target": target,
            "shared": target_parent._modules[target_name] is source_parent._modules[source_name],
        }

    def persist(self, plan):
        return {"schema_version": 1, **plan}

    def rollback(self, model, plan):
        if self._previous is not None:
            parent, name, module = self._previous
            parent._modules[name] = module
            self._previous = None
        return {"restored": True, "target": plan["target"]}
