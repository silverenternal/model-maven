"""Route source edit plans to compatible transactional handlers."""

from dataclasses import dataclass
from typing import Callable

from ..protocol import UserError


@dataclass(frozen=True)
class SourceEditDispatchRuntime:
    source_set_candidate: Callable[[dict, dict], dict]
    source_sequential_candidate: Callable[[dict, dict], dict]
    source_set_transaction_candidate: Callable[[dict, dict], dict]
    source_sequential_transaction_candidate: Callable[[dict, dict], dict]
    starter_migration_candidate: Callable[[dict, dict], dict]


def source_edit_candidate(runtime, request):
    edit = request.get("edit")
    if not isinstance(edit, dict) or edit.get("action") not in {
        "set", "add", "replace", "remove", "transaction", "migrate_starter"
    }:
        raise UserError("SOURCE_EDIT_UNSUPPORTED", "源码处理器不支持此操作。")
    if edit.get("action") == "migrate_starter":
        return runtime.starter_migration_candidate(request, edit)
    if edit.get("action") == "transaction":
        operations = edit.get("operations")
        if isinstance(operations, list) and operations:
            actions = {
                operation.get("action")
                for operation in operations
                if isinstance(operation, dict)
            }
            if len(actions) == 1 and actions == {"set"} and all(
                isinstance(item, dict) for item in operations
            ):
                return runtime.source_set_transaction_candidate(request, edit)
            if actions and actions <= {"add", "replace", "remove"} and all(
                isinstance(item, dict) for item in operations
            ):
                return runtime.source_sequential_transaction_candidate(request, edit)
            raise UserError(
                "SOURCE_EDIT_UNSUPPORTED",
                "一个源码原子计划必须全部使用 set，或全部使用 Sequential add/replace/remove。",
                "不要把 set 与结构操作混在同一计划中；结构操作可使用内置 Layer 或可导入的 Python 工厂。",
            )
        return runtime.source_set_transaction_candidate(request, edit)
    if edit.get("action") != "set":
        return runtime.source_sequential_candidate(request, edit)
    return runtime.source_set_candidate(request, edit)
