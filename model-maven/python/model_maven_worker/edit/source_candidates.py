"""Candidate generation for direct source CRUD on native module containers."""

import ast
import difflib
import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError
from ..model.structure import shared_state_groups
from .layers import _external_shared_state, old_state_names, unverified_state_storage
from .source_ast import source_argument_insertions


@dataclass(frozen=True)
class SourceCandidateRuntime:
    load_model: Callable[[dict], tuple]
    source_manifest: Callable[[str], dict]
    entrypoint: Callable[[str], Callable[..., Any]]
    normalize_path: Callable[[str], str]
    resolve_module: Callable[[Any, str], Any]
    module_entries: Callable[[Any], tuple]
    source_locations: Callable[[], dict]
    source_site_counts: Callable[[], dict]
    torch: Any
    call_name: Callable[[Any], str | None]
    unique_source_alias: Callable[[Any, str], str]
    source_factory_expression: Callable[..., tuple]
    source_module_expression: Callable[..., str]
    source_sequential_rebind: Callable[..., bytes]
    new_layer: Callable[[dict], Any]
    probe: Callable[[Any, dict], dict]
    inspect_model: Callable[[Any], dict]
    editable_params: Callable[[Any], dict | None]
    apply_edit: Callable[[Any, dict], dict]
    source_editable_fields: Callable[[Any], dict]
    has_module_state: Callable[[Any], bool]
    replace_module: Callable[[Any, str, Any], tuple]
    ast_offset: Callable[[list, int, int], int]
    apply_source_set_candidate: Callable[[Any, dict, Any], dict]


def source_sequential_candidate(runtime, request, edit):
    if request.get("variant") is not None:
        raise UserError("SOURCE_VARIANT_CONFLICT", "源码编辑不能应用在已有结构变体上。")
    model, origin, _, _ = runtime.load_model({**request, "variant": None})
    action = edit["action"]
    if action == "add":
        target_path = runtime.normalize_path(edit.get("parent"))
        container = runtime.resolve_module(model, target_path)
        old = None
        parent_path = target_path
    else:
        target_path = runtime.normalize_path(edit.get("path"))
        old = runtime.resolve_module(model, target_path)
        if old is model:
            raise UserError("ROOT_EDIT_UNSUPPORTED", "不能通过源码 Layer 操作修改根模型。")
        parent_path, name = target_path.rsplit(".", 1)
        container = runtime.resolve_module(model, parent_path)
    container_type = type(container).__name__
    if type(container) not in (runtime.torch.nn.Sequential, runtime.torch.nn.ModuleList, runtime.torch.nn.ModuleDict):
        raise UserError(
            "CONTAINER_UNSUPPORTED",
            "源码 add/replace/remove 目前只支持直接赋值构造的 nn.Sequential、nn.ModuleList 或 nn.ModuleDict。",
            "自定义容器或分支接线请使用持久变体或 Layer Edit Provider。",
        )
    modules, aliases = runtime.module_entries(model)
    if len(aliases.get(id(container), [])) > 1:
        raise UserError(
            "SOURCE_SHARED_CONTAINER_UNSUPPORTED",
            f"容器 {parent_path} 有多个共享引用，重建会改变引用关系。",
            "先在源码中明确拆分或保留共享容器，再重试。",
        )
    if old is not None and action in {"replace", "remove"} and len(aliases.get(id(old), [])) > 1:
        raise UserError(
            "SOURCE_SHARED_LAYER_UNSUPPORTED",
            f"Layer {target_path} 有共享别名，不能只改写其中一个源码位置。",
            "先明确调整共享关系，或使用 variant/provider 处理。",
        )
    impacted_shared_state = _external_shared_state(model, old, aliases.get(id(old), [])) if old is not None else []
    site = runtime.source_locations().get(id(container))
    if not site:
        raise UserError(
            "SOURCE_LOCATION_UNAVAILABLE",
            f"无法定位 {parent_path} 的 nn.{container_type} 构造代码。",
            f"需要直接赋值构造的项目内 nn.{container_type}；源码保持不变。",
        )
    site_key = (
        type(container).__module__, type(container).__qualname__,
        site.get("source_path"), site.get("line"),
    )
    same_site = [
        module for module in modules.values()
        if type(module) is type(container)
        and (candidate := runtime.source_locations().get(id(module))) is not None
        and candidate.get("source_path") == site.get("source_path")
        and candidate.get("line") == site.get("line")
    ]
    if len(same_site) > 1 or runtime.source_site_counts().get(site_key, 0) > 1:
        raise UserError(
            "SOURCE_CALLSITE_SHARED",
            f"第 {site['line']} 行可能构造多个 Sequential 实例，无法唯一确定源码重建范围。",
            "将目标容器移到只执行一次的独立构造行，或使用 --target variant。",
        )
    source_path = Path(site.get("source_path", site.get("file", "")))
    root = Path(origin["project_root"]).resolve()
    try:
        lexical_path = Path(os.path.abspath(source_path))
        relative = lexical_path.relative_to(root)
        resolved = lexical_path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise UserError("SOURCE_PATH_UNSUPPORTED", "Sequential 构造文件不在当前项目目录内。", "只允许编辑项目内的 Python 源码。") from error
    ignored = {".venv", "venv", "site-packages", "target", ".git", "__pycache__", ".model-maven"}
    if resolved.suffix != ".py" or any(part in ignored for part in relative.parts):
        raise UserError("SOURCE_PATH_UNSUPPORTED", "Sequential 构造文件不属于可编辑的项目 Python 源码。")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise UserError("SOURCE_PATH_UNSUPPORTED", "源码编辑拒绝经过符号链接的路径。")
    try:
        source_bytes = resolved.read_bytes()
        source = source_bytes.decode("utf-8")
        tree = ast.parse(source, filename=str(relative))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise UserError("SOURCE_PARSE_FAILED", f"无法安全解析构造文件：{error}", "源码保持不变；首批处理器要求 UTF-8 且当前文件语法有效。") from error
    if len(source_bytes) > 16 * 1024 * 1024:
        raise UserError("SOURCE_TOO_LARGE", "源码文件超过 16 MiB，暂不支持自动编辑。")
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and node.lineno == site["line"] and runtime.call_name(node) == container_type
    ]
    if len(calls) != 1:
        raise UserError(
            "SOURCE_CALLSITE_AMBIGUOUS",
            f"第 {site['line']} 行找到 {len(calls)} 个 Sequential 构造调用。",
            "将目标 Sequential 单独放在一个赋值语句中，再重试。",
        )
    call = calls[0]
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is call
    ]
    if len(assignments) != 1:
        raise UserError(
            "SOURCE_CALLSITE_UNSUPPORTED",
            "Sequential 必须直接出现在唯一赋值语句右侧。",
            "将 Sequential 单独赋值给变量或对象属性，再执行源码 CRUD。",
        )
    statement = assignments[0]
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1:
            raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "不支持多目标 Sequential 赋值。")
        assignment_target = statement.targets[0]
    else:
        assignment_target = statement.target
    original_items = list(container._modules.items())
    source_target = ast.unparse(assignment_target)
    source_items = []
    for key, _ in original_items:
        if container_type == "ModuleList":
            expression = f"{source_target}[{int(key)}]"
        elif container_type == "ModuleDict":
            expression = f"{source_target}[{key!r}]"
        else:
            expression = f"{source_target}._modules[{key!r}]"
        source_items.append((key, expression))
    torch_alias = runtime.unique_source_alias(tree, "_m_model_torch")
    initialization = None
    if action in {"add", "replace"}:
        layer_type = edit.get("layer_type")
        component = edit.get("component")
        parameters = edit.get("params", {})
        seed_payload = json.dumps(
            [relative.as_posix(), action, target_path, layer_type, component, parameters,
             hashlib.sha256(source_bytes).hexdigest()],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        initialization_seed = int(hashlib.sha256(seed_payload).hexdigest()[:16], 16) & ((1 << 63) - 1)
        with runtime.torch.random.fork_rng(devices=[]):
            runtime.torch.random.default_generator.manual_seed(initialization_seed)
            candidate_module = runtime.new_layer({"layer_type": layer_type, "component": component, "params": parameters})
        new_module_name = runtime.unique_source_alias(tree, "_m_model_layer")
        expression = new_module_name
        imports = []
        if component:
            factory_expression, imports = runtime.source_factory_expression(component, parameters, tree, 0)
        else:
            factory_expression = runtime.source_module_expression(layer_type, parameters, torch_alias)
        reference = edit.get("component_ref")
        component_note = None
        if isinstance(reference, dict):
            component_note = (
                f"Model-Maven component {reference.get('id')}@{reference.get('version')} "
                f"(manifest sha256 {reference.get('sha256')})"
            )
            dependencies = reference.get("dependencies", [])
            if dependencies:
                dependency_note = ", ".join(
                    f"{item.get('id')}@{item.get('version')} sha256={item.get('sha256')}"
                    for item in dependencies
                )
                component_note += f"; resolved dependencies: {dependency_note}"
        initialization = {
            "seed": initialization_seed,
            "name": new_module_name,
            "expression": factory_expression,
            "imports": imports,
            "component_note": component_note,
        }
        if action == "add":
            name = edit.get("name")
            before, after = edit.get("before"), edit.get("after")
            anchor = before or after
            if before and after:
                raise UserError("POSITION_INVALID", "before 与 after 不能同时使用。")
            if container_type == "ModuleList":
                if name not in (None, ""):
                    raise UserError(
                        "LAYER_NAME_UNSUPPORTED",
                        "ModuleList 使用数字索引命名；添加时省略 name。",
                        "before/after 使用子层索引，例如 0 或 model.blocks.0。",
                    )
                if anchor:
                    anchor = str(anchor).rsplit(".", 1)[-1]
                    if anchor not in container._modules:
                        raise UserError("ANCHOR_NOT_FOUND", f"ModuleList 中找不到索引：{anchor}")
            elif container_type == "ModuleDict":
                if not isinstance(name, str) or not name.isidentifier() or "." in name or name in container._modules:
                    raise UserError("LAYER_NAME_INVALID", "ModuleDict 分支名必须是未使用的 Python 标识符，且不能包含点号。")
                if anchor:
                    anchor = str(anchor).rsplit(".", 1)[-1]
                    if anchor not in container._modules:
                        raise UserError("ANCHOR_NOT_FOUND", f"ModuleDict 中找不到分支：{anchor}")
            else:
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or hasattr(container, name):
                    raise UserError("LAYER_NAME_INVALID", "新层名称必须是唯一标识符，不能与容器属性冲突。")
                if anchor and anchor not in container._modules:
                    raise UserError("ANCHOR_NOT_FOUND", f"容器中找不到位置：{anchor}")
            index = next((i for i, (key, _) in enumerate(original_items) if key == anchor), len(original_items))
            if after and anchor:
                index += 1
            source_items.insert(index, (name or "", expression))
            original_items.insert(index, (name or "", candidate_module))
            if container_type == "ModuleList":
                source_items = [(str(i), value) for i, (_, value) in enumerate(source_items)]
                original_items = [(str(i), child) for i, (_, child) in enumerate(original_items)]
                name = str(index)
            effects = {"action": "add", "path": parent_path, "name": name, "initialized": list(candidate_module.state_dict())}
            layer_label = f"{parent_path}.{name}"
        else:
            if old is None:
                raise UserError("LAYER_NOT_FOUND", f"找不到 Layer：{target_path}")
            _, name = target_path.rsplit(".", 1)
            previous_parameter = next(old.parameters(), None)
            if previous_parameter is not None and previous_parameter.is_floating_point():
                candidate_module.to(dtype=previous_parameter.dtype)
            candidate_module.train(old.training)
            source_items = [
                (key, expression if key == name else value)
                for key, value in source_items
            ]
            original_items = [
                (key, candidate_module if key == name else child)
                for key, child in original_items
            ]
            effects = {"action": "replace", "path": target_path, "initialized": list(candidate_module.state_dict())}
            layer_label = target_path
    else:
        if old is None:
            raise UserError("LAYER_NOT_FOUND", f"找不到 Layer：{target_path}")
        _, name = target_path.rsplit(".", 1)
        strategy = edit.get("strategy", "remove")
        if strategy == "remove":
            source_items = [(key, value) for key, value in source_items if key != name]
            original_items = [(key, child) for key, child in original_items if key != name]
            if container_type == "ModuleList":
                source_items = [(str(i), value) for i, (_, value) in enumerate(source_items)]
                original_items = [(str(i), child) for i, (_, child) in enumerate(original_items)]
            effects = {"action": "remove", "path": target_path, "strategy": "remove", "initialized": []}
        elif strategy == "bypass":
            identity = runtime.torch.nn.Identity()
            identity.train(old.training)
            source_items = [
                (key, f"{torch_alias}.nn.Identity()" if key == name else value)
                for key, value in source_items
            ]
            original_items = [
                (key, identity if key == name else child)
                for key, child in original_items
            ]
            effects = {"action": "remove", "path": target_path, "strategy": "bypass", "initialized": []}
        else:
            raise UserError("STRATEGY_INVALID", "策略应为 remove 或 bypass。")
        layer_label = target_path
    if old is not None:
        effects["old_state_not_copied"] = old_state_names(old)
    if old is not None and (unknown := unverified_state_storage(old)):
        effects["unverified_old_state_storage"] = unknown
    if impacted_shared_state:
        effects["removed_candidate_external_state_sharing"] = [
            {key: value for key, value in item.items() if key != "tensor"}
            for item in impacted_shared_state
        ]
    container._modules.clear()
    container._modules.update(original_items)
    candidate_bytes = runtime.source_sequential_rebind(
        source_bytes, tree, statement, assignment_target, source_items, initialization,
        container_type=container_type,
    )
    try:
        candidate_source = candidate_bytes.decode("utf-8")
        compile(candidate_source, str(resolved), "exec")
    except (UnicodeDecodeError, SyntaxError) as error:
        raise UserError("SOURCE_PATCH_INVALID", f"源码补丁未通过语法检查：{error}", "原文件保持不变。") from error
    validation = {"construction": "passed", "syntax": "passed", "forward": "not_run"}
    if request.get("batch"):
        observed = runtime.probe(model, {**request, "_disable_evidence_cache": True})
        validation.update({"forward": "passed", "outputs": observed["outputs"], "mode": observed["observed_mode"], "forward_evidence": "executed"})
        if "batch_content" in observed:
            validation["batch_content"] = observed["batch_content"]
        if "batch_snapshot" in observed:
            validation["batch_snapshot"] = observed["batch_snapshot"]
    elif request.get("require_forward"):
        raise UserError("BATCH_REQUIRED", "源码计划要求前向验证，但没有提供批次。", "使用 --batch module:factory。")
    relative_name = relative.as_posix()
    diff = "".join(difflib.unified_diff(
        source.splitlines(keepends=True), candidate_source.splitlines(keepends=True),
        fromfile=relative_name, tofile=relative_name + " (candidate)",
    ))
    return {
        "source_patch": {
            "path": relative_name, "line": site["line"], "layer": layer_label,
            "before_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "after_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "unified_diff": diff, "contents": candidate_source,
        },
        "validation": validation,
        "shared_state_evidence": {"scope": "in_memory_candidate_object_storage_v2", "groups": shared_state_groups(model)},
        "edit_effects": [effects],
        "modules": runtime.inspect_model(model)["modules"],
        "note": f"{container_type} CRUD 重建直接赋值的原生容器，保留未修改子层对象与参数状态；ModuleList 按顺序重编号，ModuleDict 保留分支名与顺序，新增/替换层使用确定性隔离随机种子。提交后以新 worker 重建并按需前向，失败自动回滚。",
    }


def source_sequential_transaction_state(runtime, model, root, container_path, container, namespace):
    container_type = type(container).__name__
    if type(container) not in (runtime.torch.nn.Sequential, runtime.torch.nn.ModuleList, runtime.torch.nn.ModuleDict):
        raise UserError(
            "CONTAINER_UNSUPPORTED",
            f"源码结构计划的容器 {container_path} 不是原生 nn.Sequential、nn.ModuleList 或 nn.ModuleDict。",
            "当前源码结构计划支持直接赋值构造的 nn.Sequential/nn.ModuleList/nn.ModuleDict；自定义容器请用 Layer Edit Provider。",
        )
    modules, aliases = runtime.module_entries(model)
    if len(aliases.get(id(container), [])) > 1:
        raise UserError(
            "SOURCE_SHARED_CONTAINER_UNSUPPORTED",
            f"容器 {container_path} 有多个共享引用，重建会改变引用关系。",
            "先在源码中明确拆分或保留共享容器，再重试。",
        )
    site = runtime.source_locations().get(id(container))
    if not site:
        raise UserError(
            "SOURCE_LOCATION_UNAVAILABLE",
            f"无法定位 {container_path} 的 nn.{container_type} 构造代码。",
            f"需要直接赋值构造的项目内 nn.{container_type}；源码保持不变。",
        )
    site_key = (
        type(container).__module__, type(container).__qualname__,
        site.get("source_path"), site.get("line"),
    )
    same_site = [
        module for module in modules.values()
        if type(module) is type(container)
        and (candidate := runtime.source_locations().get(id(module))) is not None
        and candidate.get("source_path") == site.get("source_path")
        and candidate.get("line") == site.get("line")
    ]
    if len(same_site) > 1 or runtime.source_site_counts().get(site_key, 0) > 1:
        raise UserError(
            "SOURCE_CALLSITE_SHARED",
            f"第 {site['line']} 行可能构造多个 Sequential 实例，无法唯一确定源码重建范围。",
            "将目标容器移到只执行一次的独立构造行，或使用 --target variant。",
        )
    source_path = Path(site.get("source_path", site.get("file", "")))
    try:
        lexical_path = Path(os.path.abspath(source_path))
        relative = lexical_path.relative_to(root)
        resolved = lexical_path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise UserError(
            "SOURCE_PATH_UNSUPPORTED",
            f"{container_path} 的构造文件不在当前项目目录内。",
            "只允许编辑项目内的 Python 源码。",
        ) from error
    ignored = {".venv", "venv", "site-packages", "target", ".git", "__pycache__", ".model-maven"}
    if resolved.suffix != ".py" or any(part in ignored for part in relative.parts):
        raise UserError("SOURCE_PATH_UNSUPPORTED", "Sequential 构造文件不属于可编辑的项目 Python 源码。")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise UserError("SOURCE_PATH_UNSUPPORTED", "源码编辑拒绝经过符号链接的路径。")
    try:
        file_size = resolved.stat().st_size
        if file_size > 16 * 1024 * 1024:
            raise UserError("SOURCE_TOO_LARGE", "源码文件超过 16 MiB，暂不支持自动编辑。")
        source_bytes = resolved.read_bytes()
        source = source_bytes.decode("utf-8")
        tree = ast.parse(source, filename=relative.as_posix())
    except UserError:
        raise
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise UserError(
            "SOURCE_PARSE_FAILED",
            f"无法解析 {container_path} 的 Sequential 构造文件：{error}",
            "源码保持不变；处理器要求 UTF-8 且当前文件语法有效。",
        ) from error
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.lineno == site["line"]
        and runtime.call_name(node) == container_type
    ]
    if len(calls) != 1:
        raise UserError(
            "SOURCE_CALLSITE_AMBIGUOUS",
            f"第 {site['line']} 行找到 {len(calls)} 个 Sequential 构造调用。",
            "将 Sequential 单独放在一个赋值语句中，再重试。",
        )
    call = calls[0]
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is call
    ]
    if len(assignments) != 1:
        raise UserError(
            "SOURCE_CALLSITE_UNSUPPORTED",
            "Sequential 必须直接出现在唯一赋值语句右侧。",
            "将 Sequential 单独赋值给变量或对象属性，再执行源码 CRUD。",
        )
    statement = assignments[0]
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1:
            raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "不支持多目标 Sequential 赋值。")
        assignment_target = statement.targets[0]
    else:
        assignment_target = statement.target
    if not isinstance(assignment_target, (ast.Name, ast.Attribute)):
        raise UserError(
            "SOURCE_CALLSITE_UNSUPPORTED",
            "Sequential 需要赋给局部变量或对象属性。",
            "先将容器单独赋值给变量或对象属性，再执行源码 CRUD。",
        )
    source_target = ast.unparse(assignment_target)
    source_items = []
    for name, _ in container._modules.items():
        if container_type == "ModuleList":
            expression = f"{source_target}[{int(name)}]"
        elif container_type == "ModuleDict":
            expression = f"{source_target}[{name!r}]"
        else:
            expression = f"{source_target}._modules[{name!r}]"
        source_items.append((name, expression))
    namespace_suffix = f"_{namespace}" if namespace else ""
    return {
        "container": container,
        "container_type": container_type,
        "container_path": container_path,
        "relative": relative,
        "resolved": resolved,
        "source_bytes": source_bytes,
        "source": source,
        "tree": tree,
        "statement": statement,
        "assignment_target": assignment_target,
        "source_items": source_items,
        "original_source_items": list(source_items),
        "torch_alias": runtime.unique_source_alias(tree, f"_m_model_torch{namespace_suffix}"),
        "namespace": namespace,
        "initializations": [],
        "operations": [],
        "line": site["line"],
    }


def source_sequential_transaction_candidate(runtime, request, edit):
    operations = edit.get("operations")
    structural_actions = {"add", "replace", "remove"}
    if (
        not isinstance(operations, list)
        or not 1 <= len(operations) <= 64
        or any(not isinstance(operation, dict) or operation.get("action") not in structural_actions for operation in operations)
    ):
        raise UserError(
            "SOURCE_EDIT_UNSUPPORTED",
            "源码结构原子计划目前只支持 1..64 项 add/replace/remove 操作。",
            "set-only 计划可以单独组合；不要把 set、custom 或组件工厂混入结构计划。",
        )
    if request.get("variant") is not None:
        raise UserError("SOURCE_VARIANT_CONFLICT", "源码编辑不能应用在已有结构变体上。")

    operation_paths = []
    for operation in operations:
        action = operation["action"]
        if action == "add":
            container_path = runtime.normalize_path(operation.get("parent"))
            affected_path = None
        else:
            affected_path = runtime.normalize_path(operation.get("path"))
            if affected_path == "model":
                raise UserError("ROOT_EDIT_UNSUPPORTED", "不能通过源码 Layer 操作修改根模型。")
            container_path = affected_path.rsplit(".", 1)[0]
        operation_paths.append((action, container_path, affected_path))
    for index, (action, _, affected_path) in enumerate(operation_paths):
        if action not in {"replace", "remove"}:
            continue
        for other_index, (_, other_container_path, _) in enumerate(operation_paths):
            if index == other_index:
                continue
            if other_container_path == affected_path or other_container_path.startswith(affected_path + "."):
                raise UserError(
                    "SOURCE_PLAN_OVERLAPPING_TARGETS",
                    f"計劃同時編輯 {affected_path} 及其內部容器 {other_container_path}。",
                    "同一計劃不能替換/刪除一個容器，同時編輯該容器或其子容器。",
                )

    model, origin, _, _ = runtime.load_model({**request, "variant": None})
    root = Path(origin["project_root"]).resolve()
    states = {}
    ordered_states = []
    source_bytes_total = 0
    source_files_seen = set()
    effects = []

    for operation_index, operation in enumerate(operations):
        action = operation["action"]
        if action == "add":
            container_path = runtime.normalize_path(operation.get("parent"))
            container = runtime.resolve_module(model, container_path)
            old = None
            target_path = container_path
        else:
            target_path = runtime.normalize_path(operation.get("path"))
            old = runtime.resolve_module(model, target_path)
            parent_path, name = target_path.rsplit(".", 1)
            container_path = parent_path
            container = runtime.resolve_module(model, parent_path)
        if type(container) not in (runtime.torch.nn.Sequential, runtime.torch.nn.ModuleList, runtime.torch.nn.ModuleDict):
            raise UserError(
                "CONTAINER_UNSUPPORTED",
                f"源码结构计划的容器 {container_path} 不是原生 nn.Sequential、nn.ModuleList 或 nn.ModuleDict。",
                "自定义容器或分支接线请使用持久变体或 Layer Edit Provider。",
            )

        state = states.get(id(container))
        if state is None:
            state = source_sequential_transaction_state(runtime,
                model, root, container_path, container, len(ordered_states)
            )
            source_name = state["relative"].as_posix()
            if source_name not in source_files_seen:
                source_bytes_total += len(state["source_bytes"])
                source_files_seen.add(source_name)
                if source_bytes_total > 64 * 1024 * 1024:
                    raise UserError("SOURCE_PLAN_TOO_LARGE", "源码原子计划涉及的原文件总量超过 64 MiB。", "减少计划涉及的文件数。")
            states[id(container)] = state
            ordered_states.append(state)

        modules, aliases = runtime.module_entries(model)
        if len(aliases.get(id(container), [])) > 1:
            raise UserError(
                "SOURCE_SHARED_CONTAINER_UNSUPPORTED",
                f"容器 {container_path} 有多个共享引用，重建会改变引用关系。",
                "先在源码中明确拆分或保留共享容器，再重试。",
            )
        if old is not None and action in {"replace", "remove"} and len(aliases.get(id(old), [])) > 1:
            raise UserError(
                "SOURCE_SHARED_LAYER_UNSUPPORTED",
                f"Layer {target_path} 有共享别名，不能只改写其中一个源码位置。",
                "先明确调整共享关系，或使用 variant/provider 处理。",
            )
        impacted_shared_state = _external_shared_state(model, old, aliases.get(id(old), [])) if old is not None else []
        source_items = state["source_items"]
        runtime_items = list(container._modules.items())
        if [name for name, _ in source_items] != [name for name, _ in runtime_items]:
            raise UserError(
                "SOURCE_PLAN_STATE_INVALID",
                f"容器 {container_path} 的计划状态与运行时 Layer 顺序不一致。",
                "拆分计划后重试；源码保持不变。",
            )

        if action in {"add", "replace"}:
            layer_type = operation.get("layer_type")
            component = operation.get("component")
            parameters = operation.get("params", {})
            if not isinstance(parameters, dict):
                raise UserError("FIELD_INVALID", f"{target_path} 的 params 必须是 JSON 对象。")
            seed_payload = json.dumps(
                [state["relative"].as_posix(), operation_index, action, target_path,
                 layer_type, component, parameters, hashlib.sha256(state["source_bytes"]).hexdigest()],
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            initialization_seed = int(hashlib.sha256(seed_payload).hexdigest()[:16], 16) & ((1 << 63) - 1)
            try:
                with runtime.torch.random.fork_rng(devices=[]):
                    runtime.torch.random.default_generator.manual_seed(initialization_seed)
                    candidate_module = runtime.new_layer({
                        "layer_type": layer_type, "component": component, "params": parameters,
                    })
            except UserError:
                raise
            except Exception as error:
                raise UserError("LAYER_CONSTRUCTION_FAILED", f"无法创建 {layer_type}：{error}") from error
            alias = runtime.unique_source_alias(state["tree"], f"_m_model_layer_{operation_index}")
            imports = []
            if component:
                factory_expression, imports = runtime.source_factory_expression(
                    component, parameters, state["tree"], operation_index,
                )
            else:
                factory_expression = runtime.source_module_expression(
                    layer_type, parameters, state["torch_alias"],
                )
            reference = operation.get("component_ref")
            component_note = None
            if isinstance(reference, dict):
                component_note = (
                    f"Model-Maven component {reference.get('id')}@{reference.get('version')} "
                    f"(manifest sha256 {reference.get('sha256')})"
                )
                dependencies = reference.get("dependencies", [])
                if dependencies:
                    dependency_note = ", ".join(
                        f"{item.get('id')}@{item.get('version')} sha256={item.get('sha256')}"
                        for item in dependencies
                    )
                    component_note += f"; resolved dependencies: {dependency_note}"
            initialization = {
                "seed": initialization_seed,
                "name": alias,
                "expression": factory_expression,
                "imports": imports,
                "component_note": component_note,
            }
            state["initializations"].append(initialization)
            if action == "add":
                name = operation.get("name")
                before, after = operation.get("before"), operation.get("after")
                anchor = before or after
                if before and after:
                    raise UserError("POSITION_INVALID", "before 与 after 不能同时使用。")
                if state["container_type"] == "ModuleList":
                    if name not in (None, ""):
                        raise UserError(
                            "LAYER_NAME_UNSUPPORTED",
                            "ModuleList 使用数字索引命名；添加时省略 name。",
                            "before/after 使用子层索引，例如 0 或 model.blocks.0。",
                        )
                    if anchor:
                        anchor = str(anchor).rsplit(".", 1)[-1]
                        if anchor not in container._modules:
                            raise UserError("ANCHOR_NOT_FOUND", f"ModuleList 中找不到索引：{anchor}")
                elif state["container_type"] == "ModuleDict":
                    if not isinstance(name, str) or not name.isidentifier() or "." in name or name in container._modules:
                        raise UserError("LAYER_NAME_INVALID", "ModuleDict 分支名必须是未使用的 Python 标识符，且不能包含点号。")
                    if anchor:
                        anchor = str(anchor).rsplit(".", 1)[-1]
                        if anchor not in container._modules:
                            raise UserError("ANCHOR_NOT_FOUND", f"ModuleDict 中找不到分支：{anchor}")
                else:
                    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or hasattr(container, name):
                        raise UserError("LAYER_NAME_INVALID", "新层名称必须是唯一标识符，不能与容器属性冲突。")
                    if anchor and anchor not in container._modules:
                        raise UserError("ANCHOR_NOT_FOUND", f"容器中找不到位置：{anchor}")
                insertion_index = next(
                    (index for index, (key, _) in enumerate(runtime_items) if key == anchor),
                    len(runtime_items),
                )
                if after and anchor:
                    insertion_index += 1
                runtime_items.insert(insertion_index, (name or "", candidate_module))
                source_items.insert(insertion_index, (name or "", alias))
                if state["container_type"] == "ModuleList":
                    runtime_items = [(str(index), child) for index, (_, child) in enumerate(runtime_items)]
                    source_items = [(str(index), expression) for index, (_, expression) in enumerate(source_items)]
                    name = str(insertion_index)
                effect = {
                    "action": "add", "path": container_path, "name": name,
                    "initialized": list(candidate_module.state_dict()),
                }
                operation_target = f"{container_path}.{name}"
            else:
                if old is None:
                    raise UserError("LAYER_NOT_FOUND", f"找不到 Layer：{target_path}")
                _, name = target_path.rsplit(".", 1)
                previous_parameter = next(old.parameters(), None)
                if previous_parameter is not None and previous_parameter.is_floating_point():
                    candidate_module.to(dtype=previous_parameter.dtype)
                candidate_module.train(old.training)
                item_index = next((index for index, (key, _) in enumerate(runtime_items) if key == name), None)
                if item_index is None:
                    raise UserError("LAYER_NOT_FOUND", f"容器 {container_path} 中找不到 Layer：{name}")
                runtime_items[item_index] = (name, candidate_module)
                source_items[item_index] = (name, alias)
                effect = {
                    "action": "replace", "path": target_path,
                    "initialized": list(candidate_module.state_dict()),
                }
                operation_target = target_path
        else:
            if old is None:
                raise UserError("LAYER_NOT_FOUND", f"找不到 Layer：{target_path}")
            _, name = target_path.rsplit(".", 1)
            strategy = operation.get("strategy", "remove")
            item_index = next((index for index, (key, _) in enumerate(runtime_items) if key == name), None)
            if item_index is None:
                raise UserError("LAYER_NOT_FOUND", f"容器 {container_path} 中找不到 Layer：{name}")
            if strategy == "remove":
                runtime_items.pop(item_index)
                source_items.pop(item_index)
                if state["container_type"] == "ModuleList":
                    runtime_items = [(str(index), child) for index, (_, child) in enumerate(runtime_items)]
                    source_items = [(str(index), expression) for index, (_, expression) in enumerate(source_items)]
            elif strategy == "bypass":
                identity = runtime.torch.nn.Identity()
                identity.train(old.training)
                runtime_items[item_index] = (name, identity)
                source_items[item_index] = (name, f"{state['torch_alias']}.nn.Identity()")
            else:
                raise UserError("STRATEGY_INVALID", "策略应为 remove 或 bypass。")
            effect = {
                "action": "remove", "path": target_path,
                "strategy": strategy, "initialized": [],
            }
            operation_target = target_path

        if old is not None:
            effect["old_state_not_copied"] = old_state_names(old)
        if old is not None and (unknown := unverified_state_storage(old)):
            effect["unverified_old_state_storage"] = unknown
        if impacted_shared_state:
            effect["removed_candidate_external_state_sharing"] = [
                {key: value for key, value in item.items() if key != "tensor"}
                for item in impacted_shared_state
            ]
        state["source_items"] = source_items
        container._modules.clear()
        container._modules.update(runtime_items)
        state["operations"].append({"action": action, "target": operation_target})
        effects.append(effect)

    patches_by_file = {}
    for state in ordered_states:
        referenced_initializers = {
            expression for _, expression in state["source_items"]
        }
        initializations = [
            item for item in state["initializations"]
            if item["name"] in referenced_initializers
        ]
        if (
            state["source_items"] == state["original_source_items"]
            and not initializations
        ):
            continue
        _, offset, snippet = runtime.source_sequential_rebind(
            state["source_bytes"], state["tree"], state["statement"],
            state["assignment_target"], state["source_items"],
            initializations, namespace=state["namespace"], return_patch=True,
            container_type=state["container_type"],
        )
        patches_by_file.setdefault(state["relative"].as_posix(), []).append((state, offset, snippet))

    if not patches_by_file:
        raise UserError("SOURCE_NO_CHANGE", "结构计划最终没有改变任何容器。", "调整操作顺序或 Layer 参数后重试。")

    source_patches = []
    snapshot_bytes_total = 0
    preview_bytes_total = 0
    for source_name, file_states in patches_by_file.items():
        original = file_states[0][0]["source_bytes"]
        if any(state["source_bytes"] != original for state, _, _ in file_states):
            raise UserError("SOURCE_PLAN_STATE_INVALID", f"{source_name} 在计划内出现多个原始版本。")
        offsets = [offset for _, offset, _ in file_states]
        if len(offsets) != len(set(offsets)):
            raise UserError("SOURCE_PLAN_OVERLAP", f"{source_name} 中的容器赋值位置重复，未应用任何修改。")
        candidate_bytes = original
        for _, offset, snippet in sorted(file_states, key=lambda item: item[1], reverse=True):
            candidate_bytes = candidate_bytes[:offset] + snippet + candidate_bytes[offset:]
        first_state = file_states[0][0]
        try:
            candidate_source = candidate_bytes.decode("utf-8")
            compile(candidate_source, str(first_state["resolved"]), "exec")
        except (UnicodeDecodeError, SyntaxError) as error:
            raise UserError(
                "SOURCE_PATCH_INVALID",
                f"成组源码结构计划未通过语法检查：{source_name}: {error}",
                "所有原文件保持不变。",
            ) from error
        snapshot_bytes_total += len(original) + len(candidate_bytes)
        if snapshot_bytes_total > 64 * 1024 * 1024:
            raise UserError("SOURCE_PLAN_TOO_LARGE", "源码计划快照总量超过 64 MiB。", "减少计划涉及的文件数或变更范围。")
        diff = "".join(difflib.unified_diff(
            original.decode("utf-8").splitlines(keepends=True),
            candidate_source.splitlines(keepends=True),
            fromfile=source_name, tofile=source_name + " (candidate)",
        ))
        preview_bytes_total += len(candidate_bytes) + 2 * len(diff.encode("utf-8"))
        if preview_bytes_total > 24 * 1024 * 1024:
            raise UserError("SOURCE_PLAN_TOO_LARGE", "源码计划的预览内容超过 24 MiB。", "减少计划中的文件或源码变更范围。")
        source_patches.append({
            "path": source_name,
            "line": min(state["line"] for state, _, _ in file_states),
            "operations": [operation for state, _, _ in file_states for operation in state["operations"]],
            "before_sha256": hashlib.sha256(original).hexdigest(),
            "after_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "unified_diff": diff,
            "contents": candidate_source,
        })

    validation = {"construction": "passed", "syntax": "passed", "forward": "not_run"}
    if request.get("batch"):
        observed = runtime.probe(model, {**request, "_disable_evidence_cache": True})
        validation.update({"forward": "passed", "outputs": observed["outputs"], "mode": observed["observed_mode"], "forward_evidence": "executed"})
        if "batch_content" in observed:
            validation["batch_content"] = observed["batch_content"]
        if "batch_snapshot" in observed:
            validation["batch_snapshot"] = observed["batch_snapshot"]
    elif request.get("require_forward"):
        raise UserError("BATCH_REQUIRED", "源码计划要求前向验证，但没有提供批次。", "使用 --batch module:factory。")

    combined_diff = "".join(item["unified_diff"] for item in source_patches)
    primary_patch = dict(source_patches[0])
    primary_patch["unified_diff"] = combined_diff
    primary_patch.pop("contents", None)
    if len(source_patches) > 1:
        primary_patch["path"] = f"{len(source_patches)} files"
        primary_patch["line"] = None
        primary_patch["operations"] = [
            {"path": item["path"], "operations": item["operations"]}
            for item in source_patches
        ]
    return {
        "source_patch": primary_patch,
        "source_patches": source_patches,
        "validation": validation,
        "shared_state_evidence": {"scope": "in_memory_candidate_object_storage_v2", "groups": shared_state_groups(model)},
        "edit_effects": effects,
        "modules": runtime.inspect_model(model)["modules"],
        "note": "成组 Sequential/ModuleList/ModuleDict CRUD 已统一构造、编译并按需前向检查；ModuleList 按索引重编号，ModuleDict 保留分支名与顺序。提交使用可恢复事务日志与逐文件原子替换，重载/前向失败将整体回滚。文件系统不提供跨文件瞬时原子可见性。",
    }


def apply_source_set_candidate(runtime, model, operation, call):
    """Apply a source-only constructor edit to the disposable candidate model."""
    path = runtime.normalize_path(operation.get("path"))
    old = runtime.resolve_module(model, path)
    if old is model:
        raise UserError("ROOT_EDIT_UNSUPPORTED", "不能通过 Layer 操作重建根模型。")
    if runtime.editable_params(old) is not None:
        return runtime.apply_edit(model, operation)
    fields = runtime.source_editable_fields(old)
    changed = operation.get("params", {})
    if not isinstance(changed, dict) or not changed or not set(changed).issubset(fields):
        raise UserError("FIELD_EDIT_UNSUPPORTED", "该自定义 Layer 的源码参数无法安全重建。")

    try:
        signature = inspect.signature(type(old).__init__)
    except (TypeError, ValueError) as error:
        raise UserError("SOURCE_CONSTRUCTOR_UNSUPPORTED", "无法读取自定义 Layer 的构造签名。") from error
    parameters = [
        (name, parameter)
        for name, parameter in signature.parameters.items()
        if name != "self"
    ]
    fixed_positional = [
        name for name, parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    values = {}

    def expression_value(node, name):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            if hasattr(old, name):
                return getattr(old, name)
            raise UserError(
                "SOURCE_CONSTRUCTOR_UNSUPPORTED",
                f"无法在候选模型中安全重建构造参数 {name}。",
                "将该参数写成字面量，或让 Layer 将其保存在同名属性中。",
            )

    if len(call.args) > len(fixed_positional):
        raise UserError("SOURCE_CONSTRUCTOR_UNSUPPORTED", "自定义 Layer 使用了无法映射的可变位置参数。")
    for name, node in zip(fixed_positional, call.args):
        values[name] = expression_value(node, name)
    for keyword in call.keywords:
        if keyword.arg is None:
            raise UserError("SOURCE_CONSTRUCTOR_UNSUPPORTED", "无法重建 **kwargs 构造表达式。")
        values[keyword.arg] = expression_value(keyword.value, keyword.arg)

    values.update(changed)
    extra_positional = []
    has_var_keyword = False
    for name, parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if name not in values:
            if hasattr(old, name):
                values[name] = getattr(old, name)
            elif parameter.default is not inspect.Parameter.empty:
                values[name] = parameter.default
            else:
                raise UserError(
                    "SOURCE_CONSTRUCTOR_UNSUPPORTED",
                    f"无法恢复构造参数 {name} 的原值。",
                    "将该参数显式写进构造调用，再重试。",
                )

    positional_only = [
        name for name, parameter in parameters
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY
    ]
    constructor_args = [values[name] for name in positional_only]
    constructor_kwargs = {
        name: values[name]
        for name, parameter in parameters
        if name in values and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    declared_names = {name for name, parameter in parameters if parameter.kind != inspect.Parameter.VAR_KEYWORD}
    if has_var_keyword:
        constructor_kwargs.update({name: value for name, value in values.items() if name not in declared_names})

    try:
        with runtime.torch.random.fork_rng(devices=[]):
            candidate = type(old)(*constructor_args, **constructor_kwargs)
    except Exception as error:
        raise UserError(
            "SOURCE_CONSTRUCTOR_INVALID",
            f"自定义 Layer 使用更新后的构造参数无法创建：{type(error).__name__}: {error}",
        ) from error
    if type(candidate) is not type(old):
        raise UserError("SOURCE_CONSTRUCTOR_INVALID", "自定义 Layer 构造器返回了不同类型。")
    if runtime.has_module_state(candidate):
        raise UserError(
            "SOURCE_STATE_MIGRATION_REQUIRED",
            "自定义 Layer 候选包含参数或 buffer，当前源码处理器不会猜测状态迁移。",
            "为该 Layer 提供显式迁移逻辑，或使用已支持的 variant/Provider 编辑处理器。",
        )
    candidate.train(old.training)
    _, aliases = runtime.module_entries(model)
    affected_aliases = aliases[id(old)]
    runtime.replace_module(model, path, candidate)
    return {
        "action": "set",
        "path": path,
        "module_type": type(candidate).__name__,
        "source_fields": sorted(changed),
        "preserved": [],
        "initialized": [],
        "affected_aliases": affected_aliases,
        "state_migration": "not_required_parameterless_module",
    }


def source_set_transaction_candidate(runtime, request, edit):
    operations = edit.get("operations")
    if (
        not isinstance(operations, list)
        or not 1 <= len(operations) <= 64
        or any(not isinstance(operation, dict) or operation.get("action") != "set" for operation in operations)
    ):
        raise UserError(
            "SOURCE_EDIT_UNSUPPORTED",
            "源码原子计划目前只支持 1..64 项 set 操作。",
            "add/replace/remove 和组件工厂请使用 variant 目标。",
        )
    if request.get("variant") is not None:
        raise UserError("SOURCE_VARIANT_CONFLICT", "源码编辑不能应用在已有结构变体上。")
    model, origin, _, _ = runtime.load_model({**request, "variant": None})
    root = Path(origin["project_root"]).resolve()
    modules = runtime.module_entries(model)[0]
    target_ids = set()
    sources = {}
    planned = []
    source_bytes_total = 0

    for operation in operations:
        if operation.get("component_ref") is not None or operation.get("component") is not None:
            raise UserError("SOURCE_EDIT_UNSUPPORTED", "源码 set 计划不支持组件工厂字段。")
        target_path = runtime.normalize_path(operation.get("path"))
        old = runtime.resolve_module(model, target_path)
        if id(old) in target_ids:
            raise UserError(
                "SOURCE_PLAN_DUPLICATE_TARGET",
                f"源码计划多次修改同一个 Layer：{target_path}。",
                "合并同一 Layer 的参数到一项 set 操作。",
            )
        target_ids.add(id(old))
        site = runtime.source_locations().get(id(old))
        if not site:
            raise UserError(
                "SOURCE_LOCATION_UNAVAILABLE",
                f"无法定位 {target_path} 的项目构造代码。",
                "该 Layer 可能由动态工厂、外部包或未追踪控制流创建。",
            )
        same_site_instances = [
            module for module in modules.values()
            if type(module) is type(old)
            and (candidate := runtime.source_locations().get(id(module))) is not None
            and candidate.get("source_path") == site.get("source_path")
            and candidate.get("line") == site.get("line")
        ]
        site_key = (
            type(old).__module__, type(old).__qualname__, site.get("source_path"), site.get("line")
        )
        if len(same_site_instances) > 1 or runtime.source_site_counts().get(site_key, 0) > 1:
            raise UserError(
                "SOURCE_CALLSITE_SHARED",
                f"第 {site['line']} 行构造了多个 {type(old).__name__} 实例，无法唯一绑定 {target_path}。",
                "将各 Layer 构造移到不同代码行，或使用 variant 目标。",
            )
        source_path = Path(site.get("source_path", site.get("file", "")))
        try:
            lexical_path = Path(os.path.abspath(source_path))
            relative = lexical_path.relative_to(root)
            resolved = lexical_path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise UserError("SOURCE_PATH_UNSUPPORTED", f"{target_path} 的构造文件不在当前项目内。") from error
        ignored = {".venv", "venv", "site-packages", "target", ".git", "__pycache__", ".model-maven"}
        if resolved.suffix != ".py" or any(part in ignored for part in relative.parts):
            raise UserError("SOURCE_PATH_UNSUPPORTED", f"{target_path} 的构造文件不是可编辑的项目 Python 源码。")
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise UserError("SOURCE_PATH_UNSUPPORTED", "源码编辑拒绝经过符号链接的路径。")
        source_name = relative.as_posix()
        source_info = sources.get(source_name)
        if source_info is None:
            try:
                file_size = resolved.stat().st_size
                if file_size > 16 * 1024 * 1024:
                    raise UserError("SOURCE_TOO_LARGE", "源码文件超过 16 MiB，暂不支持自动编辑。")
                if source_bytes_total + file_size > 64 * 1024 * 1024:
                    raise UserError("SOURCE_PLAN_TOO_LARGE", "源码原子计划的原文件总量超过 64 MiB。", "减少计划涉及的文件数。")
                source_bytes = resolved.read_bytes()
                source_bytes_total += len(source_bytes)
                source = source_bytes.decode("utf-8")
                tree = ast.parse(source, filename=source_name)
            except UserError:
                raise
            except (OSError, UnicodeDecodeError, SyntaxError) as error:
                raise UserError("SOURCE_PARSE_FAILED", f"无法安全解析构造文件：{error}") from error
            source_info = {
                "relative": relative,
                "bytes": source_bytes,
                "source": source,
                "tree": tree,
                "lines": source_bytes.splitlines(keepends=True),
                "replacements": [],
            }
            sources[source_name] = source_info

        parameters = operation.get("params", {})
        fields = runtime.source_editable_fields(old)
        if not isinstance(parameters, dict) or not parameters or set(parameters) - fields:
            raise UserError(
                "FIELD_INVALID",
                f"{target_path} 可通过源码参数处理器修改：{', '.join(sorted(fields)) or '无可识别构造参数'}",
            )
        signature = inspect.signature(type(old).__init__)
        parameter_names = [
            name for name, parameter in signature.parameters.items()
            if name != "self" and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        calls = [
            node for node in ast.walk(source_info["tree"])
            if isinstance(node, ast.Call)
            and node.lineno == site["line"]
            and runtime.call_name(node) == type(old).__name__
        ]
        if len(calls) != 1:
            raise UserError(
                "SOURCE_CALLSITE_AMBIGUOUS",
                f"第 {site['line']} 行找到 {len(calls)} 个 {type(old).__name__} 构造调用。",
                "将 Layer 构造移到独立代码行后重试。",
            )
        call = calls[0]
        if any(isinstance(argument, ast.Starred) for argument in call.args) or any(keyword.arg is None for keyword in call.keywords):
            raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "暂不改写 *args 或 **kwargs 构造表达式。")
        insertion_values = []
        for name, value in parameters.items():
            keyword = next((item for item in call.keywords if item.arg == name), None)
            if keyword is not None:
                node = keyword.value
            elif name in parameter_names and parameter_names.index(name) < len(call.args):
                node = call.args[parameter_names.index(name)]
            else:
                insertion_values.append((name, value))
                continue
            start = runtime.ast_offset(source_info["lines"], node.lineno, node.col_offset)
            end = runtime.ast_offset(source_info["lines"], node.end_lineno, node.end_col_offset)
            source_info["replacements"].append((start, end, repr(value).encode("utf-8"), target_path))
        if insertion_values:
            for start, end, replacement in source_argument_insertions(source_info["bytes"], call, insertion_values):
                source_info["replacements"].append((start, end, replacement, target_path))
        planned.append({
            "path": target_path,
            "line": site["line"],
            "params": parameters,
            "source": source_name,
            "operation": operation,
            "call": call,
        })

    source_patches = []
    total_bytes = 0
    wire_bytes = 0
    for source_name, source_info in sources.items():
        replacements = source_info["replacements"]
        ordered_replacements = sorted(replacements, key=lambda item: (item[0], item[1]))
        for previous, current in zip(ordered_replacements, ordered_replacements[1:]):
            if current[0] < previous[1] or (current[0] == previous[0] and current[0] == current[1]):
                raise UserError("SOURCE_PLAN_OVERLAP", f"{source_name} 中的 Layer 参数补丁发生重叠，未应用任何修改。")
        candidate_bytes = source_info["bytes"]
        for start, end, replacement, _ in sorted(replacements, key=lambda item: (item[0], item[1]), reverse=True):
            candidate_bytes = candidate_bytes[:start] + replacement + candidate_bytes[end:]
        if candidate_bytes == source_info["bytes"]:
            continue
        try:
            candidate_source = candidate_bytes.decode("utf-8")
            compile(candidate_source, str(root / source_info["relative"]), "exec")
        except (UnicodeDecodeError, SyntaxError) as error:
            raise UserError("SOURCE_PATCH_INVALID", f"源码计划未通过语法检查：{source_name}: {error}", "所有原文件保持不变。") from error
        total_bytes += len(source_info["bytes"]) + len(candidate_bytes)
        if total_bytes > 64 * 1024 * 1024:
            raise UserError("SOURCE_PLAN_TOO_LARGE", "源码原子计划的快照总量超过 64 MiB。", "减少计划文件数或拆分计划。")
        diff = "".join(difflib.unified_diff(
            source_info["source"].splitlines(keepends=True), candidate_source.splitlines(keepends=True),
            fromfile=source_name, tofile=source_name + " (candidate)",
        ))
        wire_bytes += len(candidate_bytes) + 2 * len(diff.encode("utf-8"))
        if wire_bytes > 24 * 1024 * 1024:
            raise UserError(
                "SOURCE_PLAN_TOO_LARGE",
                "源码计划的预览内容超过 24 MiB。",
                "减少一次计划中涉及的文件或源码变更范围。",
            )
        source_patches.append({
            "path": source_name,
            "line": min(item["line"] for item in planned if item["source"] == source_name),
            "operations": [{"path": item["path"], "params": item["params"]} for item in planned if item["source"] == source_name],
            "before_sha256": hashlib.sha256(source_info["bytes"]).hexdigest(),
            "after_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "unified_diff": diff,
            "contents": candidate_source,
        })
    if not source_patches:
        raise UserError("SOURCE_NO_CHANGE", "源码计划没有产生变更。", "检查目标参数是否已经是请求值。")
    effects = []
    for item in planned:
        operation = {key: value for key, value in item["operation"].items() if key not in {"target", "component_ref"}}
        effect = runtime.apply_source_set_candidate(model, operation, item["call"])
        if "preserved_external_state_sharing" in effect:
            effect["candidate_external_state_sharing"] = effect.pop("preserved_external_state_sharing")
        effects.append(effect)
    validation = {"construction": "passed", "syntax": "passed", "forward": "not_run"}
    if request.get("batch"):
        observed = runtime.probe(model, {**request, "_disable_evidence_cache": True})
        validation.update({"forward": "passed", "outputs": observed["outputs"], "mode": observed["observed_mode"], "forward_evidence": "executed"})
        if "batch_content" in observed:
            validation["batch_content"] = observed["batch_content"]
        if "batch_snapshot" in observed:
            validation["batch_snapshot"] = observed["batch_snapshot"]
    elif request.get("require_forward"):
        raise UserError("BATCH_REQUIRED", "源码计划要求前向验证，但没有提供批次。", "使用 --batch module:factory。")
    combined_diff = "".join(item["unified_diff"] for item in source_patches)
    primary_patch = dict(source_patches[0])
    primary_patch["unified_diff"] = combined_diff
    primary_patch.pop("contents", None)
    if len(source_patches) > 1:
        primary_patch["path"] = f"{len(source_patches)} files"
        primary_patch["line"] = None
        primary_patch["operations"] = [
            {"path": item["path"], "operations": item["operations"]} for item in source_patches
        ]
    return {
        "source_patch": primary_patch,
        "source_patches": source_patches,
        "validation": validation,
        "shared_state_evidence": {"scope": "in_memory_candidate_object_storage_v2", "groups": shared_state_groups(model)},
        "edit_effects": effects,
        "modules": runtime.inspect_model(model)["modules"],
        "note": "多文件 set 计划已统一构造和检查；提交使用可恢复事务日志与逐文件原子替换，任一重载/前向失败将整体回滚。文件系统不提供跨文件瞬时原子可见性。",
    }
