"""Explicit, fail-closed migration of model state between structures."""

from collections.abc import Mapping

from ..protocol import UserError


def migrate_state_dict(
    torch_module, source_state, target_state, specification,
    source_alias_groups=(), target_alias_groups=(),
):
    """Return a new target state and report without mutating either input.

    Tensor copying is deliberately limited to an explicit source/target mapping
    with an exact shape and dtype match. ``semantic_reason`` is the user's
    assertion that the two parameters represent the same quantity; matching
    names and shapes alone never authorize a copy. Transformations such as
    slicing, widening, merging or splitting require a future explicit adapter.
    """
    if not isinstance(source_state, Mapping) or not isinstance(target_state, Mapping):
        raise UserError("MIGRATION_STATE_INVALID", "迁移源状态和目标状态都必须是映射。")
    if not isinstance(specification, dict):
        raise UserError("MIGRATION_PLAN_INVALID", "权重迁移计划必须是 JSON 对象。")
    allowed = {"schema_version", "preserve_compatible", "reset_target", "drop_source"}
    if set(specification) - allowed or specification.get("schema_version") != 1:
        raise UserError("MIGRATION_PLAN_INVALID", "迁移计划版本或字段无效；部分张量迁移必须使用显式适配器。")

    preserve = specification.get("preserve_compatible", [])
    reset = specification.get("reset_target", [])
    drop = specification.get("drop_source", [])
    if not isinstance(preserve, list) or not isinstance(reset, list) or not isinstance(drop, list):
        raise UserError("MIGRATION_PLAN_INVALID", "preserve/reset/drop 必须是数组。")
    if len(preserve) > 100_000 or len(reset) > 100_000 or len(drop) > 100_000:
        raise UserError("MIGRATION_PLAN_INVALID", "迁移计划项目过多。")

    source_keys = set(source_state)
    target_keys = set(target_state)
    if any(not isinstance(key, str) for key in source_keys | target_keys):
        raise UserError("MIGRATION_STATE_INVALID", "状态字典键必须是字符串。")
    if any(not isinstance(value, torch_module.Tensor) for value in source_state.values()) or any(
        not isinstance(value, torch_module.Tensor) for value in target_state.values()
    ):
        raise UserError(
            "MIGRATION_ADAPTER_REQUIRED",
            "源/目标模型包含非 Tensor 状态，当前迁移协议不支持隐式复用或重置该对象。",
            "为自定义 extra state 提供显式迁移适配器后再继续；父运行保持不变。",
        )

    source_claims = []
    target_claims = []
    planned_copies = []
    for item in preserve:
        if (
            not isinstance(item, dict)
            or set(item) != {"source_key", "target_key", "semantic_reason"}
            or not isinstance(item.get("source_key"), str)
            or not isinstance(item.get("target_key"), str)
            or not isinstance(item.get("semantic_reason"), str)
            or not item["semantic_reason"].strip()
        ):
            raise UserError(
                "MIGRATION_SEMANTICS_REQUIRED",
                "每个保留映射都必须明确填写 source_key、target_key 和 semantic_reason。",
                "形状相同不代表参数语义相同；请说明为什么该映射保持了参数含义。",
            )
        source_key, target_key = item["source_key"], item["target_key"]
        if source_key not in source_keys or target_key not in target_keys:
            raise UserError(
                "MIGRATION_KEY_UNKNOWN",
                f"迁移映射引用了不存在的状态键：{source_key} → {target_key}。",
            )
        source_claims.append(source_key)
        target_claims.append(target_key)
        source_value, target_value = source_state[source_key], target_state[target_key]
        if not isinstance(source_value, torch_module.Tensor) or not isinstance(
            target_value, torch_module.Tensor
        ):
            raise UserError(
                "MIGRATION_ADAPTER_REQUIRED",
                f"{source_key} → {target_key} 不是 Tensor 到 Tensor 的直接映射。",
                "自定义 extra state 或非 Tensor 状态必须由显式迁移适配器处理。",
            )
        if source_value.shape != target_value.shape or source_value.dtype != target_value.dtype:
            raise UserError(
                "MIGRATION_ADAPTER_REQUIRED",
                f"{source_key} → {target_key} 的 shape/dtype 不完全相同。",
                "当前只支持显式的一对一精确复制；切片、扩缩、合并和拆分必须由迁移适配器实现。",
                {"source_shape": list(source_value.shape), "target_shape": list(target_value.shape),
                 "source_dtype": str(source_value.dtype), "target_dtype": str(target_value.dtype)},
            )
        planned_copies.append((source_key, target_key, source_value, item["semantic_reason"].strip()))

    if len(set(source_claims)) != len(source_claims) or len(set(target_claims)) != len(target_claims):
        raise UserError("MIGRATION_MAPPING_DUPLICATE", "每个源状态和目标状态最多只能映射一次。")

    def validate_alias_groups(groups, keys, side):
        claimed = set()
        normalized = []
        if not isinstance(groups, (list, tuple)):
            raise UserError("MIGRATION_SHARED_IDENTITY_UNKNOWN", f"{side}共享参数身份信息缺失或无效。")
        for group in groups:
            if (
                not isinstance(group, (list, tuple))
                or len(group) < 2
                or any(not isinstance(key, str) or key not in keys for key in group)
                or len(set(group)) != len(group)
                or claimed.intersection(group)
            ):
                raise UserError("MIGRATION_SHARED_IDENTITY_UNKNOWN", f"{side}共享参数身份分组无效。")
            claimed.update(group)
            normalized.append(set(group))
        return normalized

    source_groups = validate_alias_groups(source_alias_groups, source_keys, "源")
    target_groups = validate_alias_groups(target_alias_groups, target_keys, "目标")
    mapping = dict(zip(source_claims, target_claims))
    source_group_by_key = {key: group for group in source_groups for key in group}
    target_group_by_key = {key: group for group in target_groups for key in group}
    for source_key, target_key in mapping.items():
        source_group = source_group_by_key.get(source_key, {source_key})
        target_group = target_group_by_key.get(target_key, {target_key})
        mapped_source_group = {key for key in source_group if key in mapping}
        mapped_target_group = {mapping[key] for key in mapped_source_group}
        mapped_target_aliases = {key for key in target_group if key in set(target_claims)}
        if mapped_source_group != source_group or mapped_target_aliases != target_group or mapped_target_group != target_group:
            raise UserError(
                "MIGRATION_SHARED_IDENTITY_MISMATCH",
                f"迁移映射 {source_key} → {target_key} 会改变共享参数身份。",
                "要保留共享权重，请为完整别名组提供一一对应映射；否则将该组整体重置/丢弃。",
                {"source_alias_group": sorted(source_group), "target_alias_group": sorted(target_group)},
            )
    if any(not isinstance(key, str) or key not in target_keys for key in reset):
        raise UserError("MIGRATION_KEY_UNKNOWN", "reset_target 中包含无效或不存在的目标状态键。")
    if any(not isinstance(key, str) or key not in source_keys for key in drop):
        raise UserError("MIGRATION_KEY_UNKNOWN", "drop_source 中包含无效或不存在的源状态键。")
    if len(set(reset)) != len(reset) or len(set(drop)) != len(drop):
        raise UserError("MIGRATION_MAPPING_DUPLICATE", "reset_target 和 drop_source 不能包含重复键。")
    if set(target_claims) & set(reset) or set(source_claims) & set(drop):
        raise UserError("MIGRATION_MAPPING_DUPLICATE", "同一状态键不能同时复制与重置，或复制与丢弃。")
    if set(target_claims) | set(reset) != target_keys:
        raise UserError("MIGRATION_TARGET_UNACCOUNTED", "每个目标状态必须明确映射或重置。")
    if set(source_claims) | set(drop) != source_keys:
        raise UserError("MIGRATION_SOURCE_UNACCOUNTED", "每个源状态必须明确映射或丢弃。")

    migrated = {
        key: value.detach().clone() if isinstance(value, torch_module.Tensor) else value
        for key, value in target_state.items()
    }
    for _, target_key, source_value, _ in planned_copies:
        migrated[target_key] = source_value.detach().to(device=migrated[target_key].device).clone()

    preserved_alias_groups = [
        {
            "source_keys": sorted(group),
            "target_keys": sorted(mapping[key] for key in group),
        }
        for group in source_groups
        if all(key in mapping for key in group)
    ]

    report = {
        "schema_version": 1,
        "preserved": [
            {"source_key": source_key, "target_key": target_key, "semantic_reason": reason}
            for source_key, target_key, _, reason in planned_copies
        ],
        "initialized": sorted(reset),
        "dropped": sorted(drop),
        "optimizer_state": "reset_required",
        "shared_identity": "preserved_for_fully_mapped_alias_groups; topology changes require reset/drop",
        "preserved_shared_alias_groups": preserved_alias_groups,
        "partial_tensor_mapping": "unsupported_without_explicit_adapter",
    }
    return migrated, report
