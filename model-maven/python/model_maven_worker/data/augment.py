"""Explicit, path-targeted transform application for user-owned training batches."""

import sys

from ..protocol import UserError
from .transforms import describe_value, run_transform_pipeline


def apply_batch_augmentations(batch, specifications, *, resolve_entrypoint, seed=0):
    """Transform only the paths the user named; labels and metadata stay untouched."""
    if specifications is None:
        return batch, []
    if not isinstance(specifications, list):
        raise UserError("AUGMENT_PIPELINE_INVALID", "训练 augment 必须是有序数组。")
    result = batch
    observations = []
    for index, stored in enumerate(specifications):
        if not isinstance(stored, dict):
            raise UserError("AUGMENT_STEP_INVALID", f"augment[{index}] 必须是对象。")
        step_id = stored.get("id")
        target = stored.get("target")
        if not isinstance(step_id, str) or not step_id:
            raise UserError("AUGMENT_STEP_INVALID", f"augment[{index}] 缺少稳定 ID。")
        if not isinstance(target, str) or not target:
            raise UserError(
                "AUGMENT_TARGET_REQUIRED",
                f"训练增强步骤 {step_id} 没有指定数据目标。",
                "使用 m-model augment set <id> --target args.0（或 kwargs.images）后重试。",
                {"step_id": step_id},
            )
        transform = _transform_specification(stored, index)
        selected = _read_path(result, target, step_id)
        transformed, trace = _apply_to_selected(
            selected,
            transform,
            resolve_entrypoint=resolve_entrypoint,
            seed=seed + index * 1_000_003,
            step_id=step_id,
        )
        result = _replace_path(result, target.split("."), transformed, step_id)
        observations.append({
            "id": step_id,
            "type": stored.get("type"),
            "target": target,
            "input": trace["input"],
            "output": trace["outputs"][0] if trace["outputs"] else describe_value(transformed),
            "sample_count": len(trace["outputs"]),
        })
    return result, observations


def _transform_specification(stored, index):
    kind = stored.get("type")
    parameters = stored.get("parameters", {})
    if not isinstance(kind, str) or not isinstance(parameters, dict):
        raise UserError("AUGMENT_STEP_INVALID", f"augment[{index}] 缺少 type 或 parameters 对象。")
    result = {"type": kind}
    if kind == "custom":
        result["entrypoint"] = stored.get("entrypoint")
        result["parameters"] = parameters
    else:
        result.update(parameters)
    return result


def _read_path(value, path, step_id):
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise UserError("AUGMENT_TARGET_NOT_FOUND", f"步骤 {step_id} 的目标字段 {path!r} 不存在。")
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise UserError("AUGMENT_TARGET_NOT_FOUND", f"步骤 {step_id} 的目标索引 {path!r} 越界。")
            current = current[index]
        else:
            raise UserError("AUGMENT_TARGET_INVALID", f"步骤 {step_id} 无法沿 {path!r} 访问该 batch 值。")
    return current


def _replace_path(value, parts, replacement, step_id):
    if not parts:
        return replacement
    part, rest = parts[0], parts[1:]
    if isinstance(value, dict):
        if part not in value:
            raise UserError("AUGMENT_TARGET_NOT_FOUND", f"步骤 {step_id} 的目标字段 {part!r} 不存在。")
        updated = dict(value)
        updated[part] = _replace_path(value[part], rest, replacement, step_id)
        return updated
    if isinstance(value, list) and part.isdigit():
        index = int(part)
        if index >= len(value):
            raise UserError("AUGMENT_TARGET_NOT_FOUND", f"步骤 {step_id} 的目标索引 {part!r} 越界。")
        updated = list(value)
        updated[index] = _replace_path(value[index], rest, replacement, step_id)
        return updated
    if isinstance(value, tuple) and part.isdigit():
        index = int(part)
        if index >= len(value):
            raise UserError("AUGMENT_TARGET_NOT_FOUND", f"步骤 {step_id} 的目标索引 {part!r} 越界。")
        updated = list(value)
        updated[index] = _replace_path(value[index], rest, replacement, step_id)
        return tuple(updated)
    raise UserError("AUGMENT_TARGET_INVALID", f"步骤 {step_id} 无法沿目标路径更新该 batch 值。")


def _apply_to_selected(value, specification, *, resolve_entrypoint, seed, step_id):
    torch = _torch_module(value)
    if torch is not None and value.ndim == 4:
        transformed = []
        observations = []
        for index, sample in enumerate(value.unbind(0)):
            result = run_transform_pipeline(
                [specification], sample,
                resolve_entrypoint=resolve_entrypoint,
                seed=seed + index,
            )
            transformed.append(result["output"])
            observations.append(result["output_port"])
        try:
            return torch.stack(transformed, dim=0), {"input": describe_value(value), "outputs": observations}
        except Exception as error:
            raise UserError("AUGMENT_BATCH_OUTPUT_INVALID", f"步骤 {step_id} 的样本输出无法重新组批：{error}") from error
    if isinstance(value, (list, tuple)) and value and all(_is_sample(item) for item in value):
        transformed = []
        observations = []
        for index, sample in enumerate(value):
            result = run_transform_pipeline(
                [specification], sample,
                resolve_entrypoint=resolve_entrypoint,
                seed=seed + index,
            )
            transformed.append(result["output"])
            observations.append(result["output_port"])
        output = tuple(transformed) if isinstance(value, tuple) else transformed
        return output, {"input": describe_value(value), "outputs": observations}
    result = run_transform_pipeline(
        [specification], value,
        resolve_entrypoint=resolve_entrypoint,
        seed=seed,
    )
    return result["output"], {"input": result["steps"][0]["input"], "outputs": [result["output_port"]]}


def _torch_module(value):
    # Tensor subclasses can be defined in user or third-party modules. An
    # existing tensor already implies PyTorch was imported, so inspect the
    # active runtime rather than importing it for unrelated batch values.
    torch = sys.modules.get("torch")
    tensor_type = getattr(torch, "Tensor", None) if torch is not None else None
    return torch if isinstance(tensor_type, type) and isinstance(value, tensor_type) else None


def _is_sample(value):
    if _torch_module(value) is not None:
        return True
    return type(value).__module__.startswith("PIL.")
