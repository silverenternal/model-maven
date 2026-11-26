"""Variant edit planning, history replay and candidate validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import secrets

from ..protocol import UserError


def _persist_initialization_seed(runtime, operation, base_seed, history_index, operation_index=None):
    operation = dict(operation)
    if "initialization_seed" in operation:
        value = operation["initialization_seed"]
        if type(value) is not int or not 0 <= value < 2**64:
            raise UserError("EDIT_PLAN_INVALID", "Layer 初始化种子必须是 0 到 2^64-1 的整数。")
        return operation
    identity = {
        "base_seed": base_seed,
        "history_index": history_index,
        "operation": operation,
    }
    if operation_index is not None:
        identity["operation_index"] = operation_index
    operation["initialization_seed"] = int(runtime.digest(identity)[:16], 16)
    return operation


@dataclass(frozen=True)
class EditPlanRuntime:
    source_edit_candidate: Callable[[dict], dict]
    validate_variant: Callable[[dict], None]
    valid_edit_operation: Callable[[dict], bool]
    digest: Callable[[Any], str]
    load_model: Callable[[dict], tuple]
    verify_component_references: Callable[..., None]
    apply_history_item: Callable[[Any, dict], list]
    probe: Callable[[Any, dict], dict]
    source_manifest: Callable[[str], dict]
    component_operations: Callable[[dict], Iterable]
    verify_manifest: Callable[[dict], None]
    torch_version: str
    inspect_model: Callable[[Any], dict]


def edit_candidate(runtime, request):
    if request.get("edit", {}).get("target", "variant") == "source":
        edit = request.get("edit", {})
        if "rng_devices" in edit or any("rng_devices" in operation for operation in edit.get("operations", []) if isinstance(operation, dict)):
            raise UserError("LAYER_RNG_SOURCE_UNSUPPORTED", "源码编辑尚不支持选择 Layer 随机设备。")
        return runtime.source_edit_candidate(request)
    previous = request.get("variant")
    requested_policy = request.get("variant_device_policy")
    if requested_policy is not None and (not isinstance(requested_policy, str) or requested_policy not in {"legacy_cpu", "factory"}):
        raise UserError("VARIANT_DEVICE_POLICY_INVALID", "不支持的变体设备策略。")
    if previous:
        runtime.validate_variant(previous)
        device_policy = previous.get("device_policy", "legacy_cpu")
        if requested_policy is not None and requested_policy != device_policy:
            raise UserError("VARIANT_DEVICE_POLICY_CONFLICT", "不能更改已有变体的设备重建策略。", "请选择新变体。")
        edits = list(previous["edits"])
        cursor = previous["cursor"]
    else:
        device_policy = requested_policy or "legacy_cpu"
        edits, cursor = [], 0
    action = request["edit"]["action"]
    if not isinstance(action, str):
        raise UserError("EDIT_PLAN_INVALID", "Layer 操作 action 必须是字符串。")
    if "rng_devices" in request["edit"] and not runtime.valid_edit_operation(request["edit"]):
        raise UserError("LAYER_RNG_DEVICES_INVALID", "Layer 随机设备声明无效或不适用于此操作。")
    if action in {"add", "set", "replace", "remove", "custom", "sharing"} and not runtime.valid_edit_operation(request["edit"]):
        raise UserError("EDIT_PLAN_INVALID", "Layer 操作字段无效或不完整。")
    if action == "undo":
        if cursor == 0:
            raise UserError("NOTHING_TO_UNDO", "没有可撤销的编辑。")
        cursor -= 1
    elif action == "redo":
        if cursor == len(edits):
            raise UserError("NOTHING_TO_REDO", "没有可重做的编辑。")
        cursor += 1
    elif action == "transaction":
        operations = request["edit"].get("operations")
        if (
            not isinstance(operations, list)
            or not 1 <= len(operations) <= 64
            or any(not runtime.valid_edit_operation(operation) for operation in operations)
        ):
            raise UserError(
                "EDIT_PLAN_INVALID",
                "原子 Layer 计划需要 1..64 项 add/set/replace/remove/custom/sharing 操作。",
            )
        edits = edits[:cursor] + [request["edit"]]
        cursor += 1
    else:
        edits = edits[:cursor] + [request["edit"]]
        cursor += 1
    if len(edits) > 128:
        raise UserError("EDIT_LIMIT_REACHED", "当前变体最多记录 128 项编辑。", "请建立新的基础版本；不会保存不可重载的描述。")
    base_request = dict(request)
    # Construct the base once and then replay the prospective history.
    if previous:
        base_variant = dict(previous)
        base_variant["cursor"] = 0
        base_variant["revision"] = runtime.digest({k: v for k, v in base_variant.items() if k != "revision"})
        base_request["variant"] = base_variant
    elif base_request.get("seed") is None:
        # A fresh variant needs a persisted seed for replay. Draw one per
        # candidate instead of assigning every new variant the same default.
        base_request["seed"] = secrets.randbits(64)
    model, origin, seed, _ = runtime.load_model(base_request)
    # Pin each structural operation's CPU generator and any explicitly
    # selected accelerator generators. Other random sources remain unpinned.
    for history_index, edit in enumerate(edits):
        if edit.get("action") == "transaction":
            operations = []
            for operation_index, operation in enumerate(edit["operations"]):
                if operation.get("action") in {"add", "replace", "custom"}:
                    operation = _persist_initialization_seed(
                        runtime, operation, seed, history_index, operation_index
                    )
                operations.append(operation)
            edit = {**edit, "operations": operations}
            edits[history_index] = edit
        elif edit.get("action") in {"add", "replace", "custom"}:
            edits[history_index] = _persist_initialization_seed(
                runtime, edit, seed, history_index
            )
    effects = []
    for index, edit in enumerate(edits[:cursor]):
        runtime.verify_component_references(edit, origin["project_root"])
        if device_policy == "factory":
            item_effects = runtime.apply_history_item(model, edit, preserve_factory_device=True)
        else:
            item_effects = runtime.apply_history_item(model, edit)
        if index == cursor - 1:
            effects = item_effects
    if action in {"undo", "redo"}:
        effects = [{"action": action, "cursor": cursor}]
    validation = {"construction": "passed", "forward": "not_run"}
    if request.get("batch"):
        try:
            observed = runtime.probe(model, request)
        except UserError as error:
            if error.code == "FORWARD_FAILED":
                raise UserError("EDIT_VALIDATION_FAILED", str(error), "原变体未提交。先用 --dry-run 查看；相关维度需要一起修改时，当前可先在源代码完成。", error.data) from error
            raise
        validation.update({"forward": "passed", "outputs": observed["outputs"], "mode": observed["observed_mode"]})
        validation["forward_evidence"] = (
            "reused" if observed.get("evidence_cache", {}).get("status") == "reused" else "executed"
        )
        if "batch_content" in observed:
            validation["batch_content"] = observed["batch_content"]
        if "batch_snapshot" in observed:
            validation["batch_snapshot"] = observed["batch_snapshot"]
    elif request.get("require_forward"):
        raise UserError("BATCH_REQUIRED", "要求前向验证，但没有提供批次。", "使用 --batch module:factory。")
    manifest = dict(previous["source_manifest"]) if previous else {}
    manifest.update(runtime.source_manifest(origin["project_root"]))
    for edit in edits:
        for _, reference in runtime.component_operations(edit):
            runtime.verify_component_references(edit, origin["project_root"])
            component_manifest = (Path(origin["project_root"]) / reference["path"]).resolve()
            manifest[str(component_manifest)] = reference["sha256"]
    runtime.verify_manifest(manifest)
    variant = {"schema_version": 1, "origin": origin, "seed": seed, "torch_version": runtime.torch_version,
               "source_manifest": manifest, "edits": edits, "cursor": cursor}
    if previous and "device_policy" in previous:
        variant["device_policy"] = previous["device_policy"]
    elif not previous and device_policy != "legacy_cpu":
        variant["device_policy"] = device_policy
    if previous and "base_rng_devices" in previous:
        variant["base_rng_devices"] = previous["base_rng_devices"]
    elif not previous and request.get("variant_base_rng_devices") is not None:
        variant["base_rng_devices"] = request["variant_base_rng_devices"]
    variant["revision"] = runtime.digest(variant)
    inspection = runtime.inspect_model(model)
    base_rng_scope = ("cpu_and_explicit_devices" if "base_rng_devices" in variant
                      else "legacy_torch_manual_seed")
    return {"variant": variant, "validation": validation, "edit_effects": effects,
            "initialization_seed_decimal": str(seed),
            "initialization_seed_source": "existing_variant" if previous else "explicit" if request.get("seed") is not None else "generated_for_candidate",
            "layer_initialization_rng_scope": "cpu_and_explicit_devices_per_operation",
            "base_initialization_rng_scope": base_rng_scope,
            "modules": inspection["modules"],
            "model_state_placement": inspection.get("model_state_placement"),
            "placement_timing": "after_candidate_validation",
            "note": "持久化的是结构配方；未保存训练权重。新增层按策略构造，替换或改参会参考旧层首个参数或 buffer 的设备与类型；输入不自动迁移。Layer 初始化隔离 CPU 及各操作显式声明的设备生成器；基础模型使用变体记录的范围或历史策略，其他设备及用户随机源不由此固定，不能保证权重一致。新变体预览若使用生成的种子，提交时可通过 --seed 使用该种子重建候选。"}
