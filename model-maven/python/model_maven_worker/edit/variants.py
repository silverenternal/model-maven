"""Validation for persisted variant histories and edit operations."""

import re
from dataclasses import dataclass
from typing import Any, Callable

from ..protocol import UserError
from .initialization_rng import validate_rng_devices


@dataclass(frozen=True)
class EditValidationRuntime:
    valid_edit_actions: set[str]
    digest: Callable[[Any], str]
    validate_model_injections: Callable[[Any], dict]


def valid_edit_operation(runtime, operation):
    if (
        not isinstance(operation, dict)
        or not isinstance(operation.get("action"), str)
        or operation["action"] not in runtime.valid_edit_actions
    ):
        return False
    action = operation["action"]
    if action in {"add", "replace"}:
        component = operation.get("component")
        layer_type = operation.get("layer_type")
        if (
            not isinstance(operation.get("params", {}), dict)
            or (component is not None and (not isinstance(component, str) or not component))
            or (layer_type is not None and (not isinstance(layer_type, str) or not layer_type))
            or (component is None and layer_type is None)
            or (component is not None and layer_type is not None)
        ):
            return False
        if action == "add":
            if not isinstance(operation.get("parent"), str) or not operation["parent"]:
                return False
            for field in ("name", "before", "after"):
                if operation.get(field) is not None and not isinstance(operation[field], str):
                    return False
            if operation.get("before") and operation.get("after"):
                return False
        elif not isinstance(operation.get("path"), str) or not operation["path"]:
            return False
    elif action == "set":
        if (
            not isinstance(operation.get("path"), str)
            or not operation["path"]
            or not isinstance(operation.get("params"), dict)
            or not operation["params"]
        ):
            return False
    elif action == "remove":
        if not isinstance(operation.get("path"), str) or not operation["path"]:
            return False
        strategy = operation.get("strategy", "remove")
        if not isinstance(strategy, str) or strategy not in {"remove", "bypass"}:
            return False
    if "initialization_seed" in operation and (
        type(operation["initialization_seed"]) is not int
        or not 0 <= operation["initialization_seed"] < 2**64
    ):
        return False
    if "rng_devices" in operation:
        if operation["action"] not in {"add", "replace"}:
            return False
        try:
            validate_rng_devices(operation["rng_devices"])
        except UserError:
            return False
    if action == "custom":
        return isinstance(operation.get("provider"), str) and isinstance(
            operation.get("payload", {}), dict
        )
    if action == "sharing":
        mode = operation.get("mode")
        if mode == "clone-selected":
            return isinstance(operation.get("path"), str) and bool(operation["path"])
        if mode == "share-selected":
            targets = operation.get("targets")
            return (
                isinstance(operation.get("source"), str)
                and bool(operation["source"])
                and isinstance(targets, list)
                and 1 <= len(targets) <= 64
                and all(isinstance(path, str) and path for path in targets)
            )
        return False
    return True


def validate_variant(runtime, variant):
    if (
        not isinstance(variant, dict)
        or type(variant.get("schema_version")) is not int
        or variant.get("schema_version") != 1
    ):
        raise UserError("VARIANT_INVALID", "不支持或无效的变体格式。")
    try:
        payload = {key: value for key, value in variant.items() if key != "revision"}
        if variant["revision"] != runtime.digest(payload):
            raise ValueError("revision")
        if (
            not isinstance(variant["edits"], list)
            or type(variant["cursor"]) is not int
            or not 0 <= variant["cursor"] <= len(variant["edits"])
        ):
            raise ValueError("cursor")
        if len(variant["edits"]) > 128:
            raise ValueError("edit limit")
        for item in variant["edits"]:
            if not isinstance(item, dict):
                raise ValueError("edit item")
            if item.get("action") == "transaction":
                operations = item.get("operations")
                if (
                    not isinstance(operations, list)
                    or not 1 <= len(operations) <= 64
                    or any(not valid_edit_operation(runtime, operation) for operation in operations)
                ):
                    raise ValueError("transaction")
            elif not valid_edit_operation(runtime, item):
                raise ValueError("edit action")
        for key in ("origin", "seed", "torch_version", "source_manifest"):
            variant[key]
        if variant.get("device_policy", "legacy_cpu") not in {"legacy_cpu", "factory"}:
            raise ValueError("device_policy")
        if "base_rng_devices" in variant:
            try:
                validate_rng_devices(variant["base_rng_devices"])
            except UserError as error:
                raise ValueError("base_rng_devices") from error
        origin = variant["origin"]
        if not isinstance(origin, dict) or not isinstance(origin.get("model"), str) or not isinstance(origin.get("params"), dict) or not isinstance(origin.get("project_root"), str):
            raise ValueError("origin")
        runtime.validate_model_injections(origin.get("model_injections", {}))
        if type(variant["seed"]) is not int or not 0 <= variant["seed"] < 2 ** 64:
            raise ValueError("seed")
        if not isinstance(variant["torch_version"], str):
            raise ValueError("torch_version")
        if not isinstance(variant["source_manifest"], dict) or not all(
            isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)
            for k, v in variant["source_manifest"].items()
        ):
            raise ValueError("source_manifest")
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("VARIANT_INVALID", f"变体完整性检查失败：{error}") from error
