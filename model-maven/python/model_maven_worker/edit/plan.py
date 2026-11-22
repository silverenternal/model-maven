"""Variant edit planning, history replay and candidate validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

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
        return runtime.source_edit_candidate(request)
    previous = request.get("variant")
    if previous:
        runtime.validate_variant(previous)
        edits = list(previous["edits"])
        cursor = previous["cursor"]
    else:
        edits, cursor = [], 0
    action = request["edit"]["action"]
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
                "原子 Layer 计划需要 1..64 项 add/set/replace/remove/custom 操作。",
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
    model, origin, seed, _ = runtime.load_model(base_request)
    # Layer factories may consume PyTorch's global RNG. Persist a stable seed
    # with each structural operation so replay reconstructs the same initial
    # weights without advancing the model's own random stream.
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
    variant["revision"] = runtime.digest(variant)
    return {"variant": variant, "validation": validation, "edit_effects": effects,
            "modules": runtime.inspect_model(model)["modules"],
            "note": "持久化的是结构配方；未保存训练权重，未接入 checkpoint 迁移。"}
