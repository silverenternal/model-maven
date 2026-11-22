"""Versioned, replayable Layer Edit Provider invocation helpers."""

import json

from ..protocol import UserError


_PROVIDER_METHODS = ("describe", "can_edit", "plan", "candidate", "persist", "rollback")
_MAX_PROVIDER_JSON_BYTES = 64 * 1024


def _json_object(value, field):
    if not isinstance(value, dict):
        raise UserError("CUSTOM_EDIT_RESULT_INVALID", f"Provider {field} 必须返回 JSON 对象。")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise UserError("CUSTOM_EDIT_RESULT_INVALID", f"Provider {field} 必须是有限 JSON 数据：{error}") from error
    if len(encoded.encode("utf-8")) > _MAX_PROVIDER_JSON_BYTES:
        raise UserError("CUSTOM_EDIT_RESULT_INVALID", f"Provider {field} 超过 64 KiB。")
    return value


def _load_provider(runtime, operation):
    provider_ref = operation.get("provider")
    if not isinstance(provider_ref, str) or ":" not in provider_ref:
        raise UserError("CUSTOM_EDIT_INVALID", "custom 操作需要 module:provider 入口。")
    module_name = provider_ref.split(":", 1)[0]
    try:
        module = __import__(module_name, fromlist=["*"])
        entry = runtime.entrypoint(provider_ref)
        version = getattr(module, "MODEL_MAVEN_LAYER_EDIT_API", None)
    except Exception as error:
        raise UserError("CUSTOM_PROVIDER_LOAD_FAILED", f"无法加载 Layer 提供者 {provider_ref}: {error}") from error

    if type(version) is not int:
        raise UserError("CUSTOM_PROVIDER_VERSION", f"{provider_ref} 必须声明整数 MODEL_MAVEN_LAYER_EDIT_API。")
    if version == 1:
        if not callable(entry):
            raise UserError("CUSTOM_PROVIDER_INVALID", "Layer Edit Provider v1 入口必须可调用。")
        return ("v1", entry)
    if version != 2:
        raise UserError("CUSTOM_PROVIDER_VERSION", f"不支持 Layer Edit Provider API {version}；当前支持 v1 和 v2。")

    try:
        provider = entry() if isinstance(entry, type) else entry
    except Exception as error:
        raise UserError("CUSTOM_PROVIDER_LOAD_FAILED", f"无法创建 Provider {provider_ref}: {error}") from error
    missing = [name for name in _PROVIDER_METHODS if not callable(getattr(provider, name, None))]
    if missing:
        raise UserError("CUSTOM_PROVIDER_INVALID", f"Provider v2 缺少方法：{', '.join(missing)}。")
    return ("v2", provider)


def apply_provider(runtime, model, operation):
    """Apply an explicitly versioned provider without accepting opaque effects."""
    if not isinstance(operation.get("payload", {}), dict):
        raise UserError("CUSTOM_EDIT_INVALID", "custom 操作 payload 必须是 JSON 对象。")
    version, provider = _load_provider(runtime, operation)
    if version == "v1":
        try:
            result = provider(model, operation)
        except UserError:
            raise
        except Exception as error:
            raise UserError("CUSTOM_EDIT_FAILED", f"自定义 Layer 编辑失败：{type(error).__name__}: {error}") from error
        return {"action": "custom", "provider": operation["provider"], "result": _json_object(result, "v1 结果")}

    try:
        description = _json_object(provider.describe(model, operation), "describe")
        decision = provider.can_edit(model, operation)
        if not isinstance(decision, dict) or decision.get("state") not in {"supported", "unsupported", "conditional"}:
            raise UserError("CUSTOM_PROVIDER_INVALID", "can_edit 必须返回 state 为 supported/unsupported/conditional 的对象。")
        if decision["state"] != "supported":
            reason = decision.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                reason = "Provider 未声明当前编辑上下文可用。"
            raise UserError("CUSTOM_EDIT_UNAVAILABLE", reason, "查看 Provider 的编辑说明并补齐明确前置条件后重试。", {"state": decision["state"], "description": description})

        if "provider_plan" in operation:
            plan = _json_object(operation["provider_plan"], "已保存计划")
        else:
            plan = _json_object(provider.plan(model, operation), "plan")
            operation["provider_plan"] = plan
        persisted = _json_object(provider.persist(plan), "persist")
        if "provider_persistence" in operation and operation["provider_persistence"] != persisted:
            raise UserError("CUSTOM_PROVIDER_PLAN_DRIFT", "Provider 对已保存计划生成了不同的持久描述；拒绝重放。")
        operation["provider_persistence"] = persisted
        result = _json_object(provider.candidate(model, plan), "candidate")
    except Exception as error:
        plan = operation.get("provider_plan")
        if isinstance(plan, dict):
            try:
                rollback_result = provider.rollback(model, plan)
                _json_object(rollback_result, "rollback")
            except Exception as rollback_error:
                raise UserError(
                    "CUSTOM_EDIT_ROLLBACK_FAILED",
                    f"Provider 编辑失败且回滚也失败：{type(rollback_error).__name__}: {rollback_error}",
                    "当前模型候选已丢弃；检查 Provider 的外部副作用并重新载入模型。",
                ) from error
        if isinstance(error, UserError):
            raise
        raise UserError("CUSTOM_EDIT_FAILED", f"自定义 Layer 编辑失败：{type(error).__name__}: {error}") from error
    return {"action": "custom", "provider": operation["provider"], "description": description, "plan": persisted, "result": result}
