"""Candidate generation for direct source CRUD on native module containers."""

import ast
import difflib
import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError
from ..model.structure import shared_state_groups
from .layers import (
    _automatic_sequential_name,
    _external_shared_state,
    _validate_layer_name,
    old_state_names,
    unverified_state_storage,
)
from .source_ast import read_bounded_python_source, source_argument_insertions, static_keyword_nodes
from ..runtime.source_paths import is_project_python_source


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
            f"第 {site['line']} 行可能构造多个 {container_type} 实例，无法唯一确定源码重建范围。",
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
        raise UserError("SOURCE_PATH_UNSUPPORTED", f"{container_type} 构造文件不在当前项目目录内。", "只允许编辑项目内的 Python 源码。") from error
    if not is_project_python_source(root, resolved):
        raise UserError("SOURCE_PATH_UNSUPPORTED", f"{container_type} 构造文件不属于可编辑的项目 Python 源码。")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise UserError("SOURCE_PATH_UNSUPPORTED", "源码编辑拒绝经过符号链接的路径。")
    try:
        source_bytes = read_bounded_python_source(resolved)
        source = source_bytes.decode("utf-8")
        tree = ast.parse(source, filename=str(relative))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise UserError("SOURCE_PARSE_FAILED", f"无法安全解析构造文件：{error}", "源码保持不变；首批处理器要求 UTF-8 且当前文件语法有效。") from error
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and node.lineno == site["line"] and runtime.call_name(node) == container_type
    ]
    if len(calls) != 1:
        raise UserError(
            "SOURCE_CALLSITE_AMBIGUOUS",
            f"第 {site['line']} 行找到 {len(calls)} 个 {container_type} 构造调用。",
            f"将目标 {container_type} 单独放在一个赋值语句中，再重试。",
        )
    call = calls[0]
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is call
    ]
    if len(assignments) != 1:
        raise UserError(
            "SOURCE_CALLSITE_UNSUPPORTED",
            f"{container_type} 必须直接出现在唯一赋值语句右侧。",
            f"将 {container_type} 单独赋值给变量或对象属性，再执行源码 CRUD。",
        )
    statement = assignments[0]
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1:
            raise UserError("SOURCE_CALLSITE_UNSUPPORTED", f"不支持多目标 {container_type} 赋值。")
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
    add_name, add_anchor, add_after = None, None, False
    if action == "add":
        add_name = edit.get("name")
        before, after = edit.get("before"), edit.get("after")
        if before and after:
            raise UserError("POSITION_INVALID", "before 与 after 不能同时使用。")
        add_anchor, add_after = before or after, bool(after)
        if container_type == "ModuleList":
            if add_name not in (None, ""):
                raise UserError(
                    "LAYER_NAME_UNSUPPORTED",
                    "ModuleList 使用数字索引命名；添加时省略 name。",
                    "before/after 使用子层索引，例如 0 或 model.blocks.0。",
                )
            if add_anchor:
                add_anchor = str(add_anchor).rsplit(".", 1)[-1]
                if add_anchor not in container._modules:
                    raise UserError("ANCHOR_NOT_FOUND", f"ModuleList 中找不到索引：{add_anchor}")
        elif container_type == "ModuleDict":
            _validate_layer_name(container, add_name)
            if add_anchor:
                add_anchor = str(add_anchor).rsplit(".", 1)[-1]
                if add_anchor not in container._modules:
                    raise UserError("ANCHOR_NOT_FOUND", f"ModuleDict 中找不到分支：{add_anchor}")
        else:
            if add_name in (None, ""):
                add_name = _automatic_sequential_name(container)
            _validate_layer_name(container, add_name)
            if add_anchor and add_anchor not in container._modules:
                raise UserError("ANCHOR_NOT_FOUND", f"容器中找不到位置：{add_anchor}")
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
            name = add_name
            index = next((i for i, (key, _) in enumerate(original_items) if key == add_anchor), len(original_items))
            if add_after and add_anchor:
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
            # The generated source does not contain a placement conversion.
            # Do not validate a converted candidate that it cannot reproduce.
            candidate_module.train(old.training)
            initialization["training_from"] = next(value for key, value in source_items if key == name)
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
                (key, f"{torch_alias}.nn.Identity().train(({value}).training)" if key == name else value)
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
    effects["source_rebind_preserves_original_construction"] = True
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
        **{key: value for key, value in runtime.inspect_model(model).items() if key in {"modules", "model_state_placement"}},
        "note": f"{container_type} CRUD 重建直接赋值的原生容器，保留未修改子层对象与参数状态；ModuleList 按顺序重编号，ModuleDict 保留分支名与顺序，新增/替换层使用确定性隔离随机种子。原容器构造代码仍会执行，被替换或删除的层仍可能先被构造；多次编辑会追加重建代码。提交后以新 worker 重建并按需前向，失败自动回滚。",
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
            f"第 {site['line']} 行可能构造多个 {container_type} 实例，无法唯一确定源码重建范围。",
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
    if not is_project_python_source(root, resolved):
        raise UserError("SOURCE_PATH_UNSUPPORTED", f"{container_type} 构造文件不属于可编辑的项目 Python 源码。")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise UserError("SOURCE_PATH_UNSUPPORTED", "源码编辑拒绝经过符号链接的路径。")
    try:
        file_size = resolved.stat().st_size
        if file_size > 16 * 1024 * 1024:
            raise UserError("SOURCE_TOO_LARGE", "源码文件超过 16 MiB，暂不支持自动编辑。")
        source_bytes = read_bounded_python_source(resolved)
        source = source_bytes.decode("utf-8")
        tree = ast.parse(source, filename=relative.as_posix())
    except UserError:
        raise
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise UserError(
            "SOURCE_PARSE_FAILED",
            f"无法解析 {container_path} 的 {container_type} 构造文件：{error}",
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
            f"第 {site['line']} 行找到 {len(calls)} 个 {container_type} 构造调用。",
            f"将 {container_type} 单独放在一个赋值语句中，再重试。",
        )
    call = calls[0]
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is call
    ]
    if len(assignments) != 1:
        raise UserError(
            "SOURCE_CALLSITE_UNSUPPORTED",
            f"{container_type} 必须直接出现在唯一赋值语句右侧。",
            f"将 {container_type} 单独赋值给变量或对象属性，再执行源码 CRUD。",
        )
    statement = assignments[0]
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1:
            raise UserError("SOURCE_CALLSITE_UNSUPPORTED", f"不支持多目标 {container_type} 赋值。")
        assignment_target = statement.targets[0]
    else:
        assignment_target = statement.target
    if not isinstance(assignment_target, (ast.Name, ast.Attribute)):
        raise UserError(
            "SOURCE_CALLSITE_UNSUPPORTED",
            f"{container_type} 需要赋给局部变量或对象属性。",
            "先将容器单独赋值给变量或对象属性，再执行源码 CRUD。",
        )
    namespace_suffix = f"_{namespace}" if namespace else ""
    original_alias = runtime.unique_source_alias(tree, f"_m_model_original_modules{namespace_suffix}")
    source_items = []
    for name, _ in container._modules.items():
        expression = f"{original_alias}[{name!r}]"
        source_items.append((name, expression))
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
        "original_alias": original_alias,
        "torch_alias": runtime.unique_source_alias(tree, f"_m_model_torch{namespace_suffix}"),
        "namespace": namespace,
        "initializations": [],
        "stages": [],
        "operations": [],
        "line": site["line"],
    }


def source_sequential_transaction_candidate(runtime, request, edit, *, shared_model=None):
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
    # Source patches are emitted once per container. An interleaved sequence
    # such as A, B, A would necessarily reorder factory calls when grouped,
    # regardless of the order in which the model constructs those containers.
    factory_groups = []
    for operation, (action, container_path, _) in zip(operations, operation_paths):
        creates_layer = action in {"add", "replace"} or (action == "remove" and operation.get("strategy") == "bypass")
        if creates_layer and (not factory_groups or factory_groups[-1] != container_path):
            factory_groups.append(container_path)
    if len(set(factory_groups)) != len(factory_groups):
        raise UserError(
            "SOURCE_PLAN_FACTORY_ORDER_UNSUPPORTED",
            "源码计划交错构造多个容器的 Layer，无法保持工厂调用顺序。",
            "将每个容器的构造操作集中排列后重新预览，或在源码中手动实现并重验。",
        )
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

    model, origin, _, _ = shared_model or runtime.load_model({**request, "variant": None})
    root = Path(origin["project_root"]).resolve()
    states = {}
    ordered_states = []
    source_bytes_total = 0
    source_files_seen = set()
    effects = []

    # Resolve the original container sites before running any candidate Layer
    # factories. This permits definite source-order conflicts to fail early.
    path_states = {}
    for _, container_path, _ in operation_paths:
        if container_path in path_states:
            continue
        container = runtime.resolve_module(model, container_path)
        state = states.get(id(container))
        if state is None:
            state = source_sequential_transaction_state(runtime,
                model, root, container_path, container, len(ordered_states))
            source_name = state["relative"].as_posix()
            if source_name not in source_files_seen:
                source_bytes_total += len(state["source_bytes"])
                source_files_seen.add(source_name)
                if source_bytes_total > 64 * 1024 * 1024:
                    raise UserError("SOURCE_PLAN_TOO_LARGE", "源码原子计划涉及的原文件总量超过 64 MiB。", "减少计划涉及的文件数。")
            states[id(container)] = state
            ordered_states.append(state)
        path_states[container_path] = state

    def direct_statement_position(state):
        # Only sibling statements in one AST list have a comparable order.
        for parent in ast.walk(state["tree"]):
            for field, children in ast.iter_fields(parent):
                if isinstance(children, list):
                    for index, child in enumerate(children):
                        if child is state["statement"]:
                            return (type(parent).__name__, getattr(parent, "lineno", 0),
                                    getattr(parent, "col_offset", 0), field), index
        return None

    for index, first_path in enumerate(factory_groups):
        first = path_states[first_path]
        for second_path in factory_groups[index + 1:]:
            second = path_states[second_path]
            if first["relative"] != second["relative"]:
                raise UserError(
                    "SOURCE_PLAN_FACTORY_ORDER_UNVERIFIABLE",
                    "跨源码文件的容器构造顺序无法静态核对。",
                    "本结构计划的新增或替换工厂尚未执行；模型构造及先前参数修改可能已有副作用。分步编辑，或手动编辑源码并重验。",
                )
            first_pos, second_pos = direct_statement_position(first), direct_statement_position(second)
            if not first_pos or not second_pos or first_pos[0] != second_pos[0]:
                raise UserError(
                    "SOURCE_PLAN_FACTORY_ORDER_UNVERIFIABLE",
                    "不同代码块中的容器构造顺序无法静态核对。",
                    "本结构计划的新增或替换工厂尚未执行；模型构造及先前参数修改可能已有副作用。分步编辑，或手动编辑源码并重验。",
                )
            if first_pos[1] > second_pos[1]:
                raise UserError(
                    "SOURCE_PLAN_FACTORY_ORDER_UNSUPPORTED",
                    "计划中的工厂顺序与同一代码块内的容器构造顺序相反。",
                    "本结构计划的新增或替换工厂尚未执行；模型构造及先前参数修改可能已有副作用。按源码顺序重新规划，或手动编辑并重验。",
                )

    for operation_index, operation in enumerate(operations):
        action = operation["action"]
        initialization = None
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
            raise UserError("SOURCE_PLAN_STATE_INVALID", "计划中的容器身份在候选操作期间发生变化。", "拆分计划或手工编辑后重验。")

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

        add_name, add_anchor, add_after = None, None, False
        if action == "add":
            add_name = operation.get("name")
            before, after = operation.get("before"), operation.get("after")
            if before and after:
                raise UserError("POSITION_INVALID", "before 与 after 不能同时使用。")
            add_anchor, add_after = before or after, bool(after)
            if state["container_type"] == "ModuleList":
                if add_name not in (None, ""):
                    raise UserError(
                        "LAYER_NAME_UNSUPPORTED",
                        "ModuleList 使用数字索引命名；添加时省略 name。",
                        "before/after 使用子层索引，例如 0 或 model.blocks.0。",
                    )
                if add_anchor:
                    add_anchor = str(add_anchor).rsplit(".", 1)[-1]
                    if add_anchor not in container._modules:
                        raise UserError("ANCHOR_NOT_FOUND", f"ModuleList 中找不到索引：{add_anchor}")
            elif state["container_type"] == "ModuleDict":
                _validate_layer_name(container, add_name)
                if add_anchor:
                    add_anchor = str(add_anchor).rsplit(".", 1)[-1]
                    if add_anchor not in container._modules:
                        raise UserError("ANCHOR_NOT_FOUND", f"ModuleDict 中找不到分支：{add_anchor}")
            else:
                if add_name in (None, ""):
                    add_name = _automatic_sequential_name(container)
                _validate_layer_name(container, add_name)
                if add_anchor and add_anchor not in container._modules:
                    raise UserError("ANCHOR_NOT_FOUND", f"容器中找不到位置：{add_anchor}")

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
                insertion_index = next(
                    (index for index, (key, _) in enumerate(runtime_items) if key == add_anchor),
                    len(runtime_items),
                )
                if add_after and add_anchor:
                    insertion_index += 1
                runtime_items.insert(insertion_index, (add_name or "", candidate_module))
                source_items.insert(insertion_index, (add_name or "", alias))
                if state["container_type"] == "ModuleList":
                    runtime_items = [(str(index), child) for index, (_, child) in enumerate(runtime_items)]
                    source_items = [(str(index), expression) for index, (_, expression) in enumerate(source_items)]
                    add_name = str(insertion_index)
                effect = {
                    "action": "add", "path": container_path, "name": add_name,
                    "initialized": list(candidate_module.state_dict()),
                }
                operation_target = f"{container_path}.{add_name}"
            else:
                if old is None:
                    raise UserError("LAYER_NOT_FOUND", f"找不到 Layer：{target_path}")
                _, name = target_path.rsplit(".", 1)
                # Keep candidate placement aligned with the generated source;
                # it has no implicit conversion to the old layer's state.
                candidate_module.train(old.training)
                item_index = next((index for index, (key, _) in enumerate(runtime_items) if key == name), None)
                if item_index is None:
                    raise UserError("LAYER_NOT_FOUND", f"容器 {container_path} 中找不到 Layer：{name}")
                initialization["training_from"] = source_items[item_index][1]
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
                previous = source_items[item_index][1]
                alias = runtime.unique_source_alias(state["tree"], f"_m_model_bypass_{operation_index}")
                initialization = {
                    "seed": None, "name": alias,
                    "expression": f"{state['torch_alias']}.nn.Identity()",
                    "training_from": previous,
                }
                state["initializations"].append(initialization)
                source_items[item_index] = (name, alias)
            else:
                raise UserError("STRATEGY_INVALID", "策略应为 remove 或 bypass。")
            effect = {
                "action": "remove", "path": target_path,
                "strategy": strategy, "initialized": [],
            }
            operation_target = target_path

        if old is not None:
            effect["old_state_not_copied"] = old_state_names(old)
        effect["source_rebind_preserves_original_construction"] = True
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
        state["stages"].append({"initialization": initialization, "items": list(source_items)})
        state["operations"].append({"action": action, "target": operation_target})
        effects.append(effect)

    patches_by_file = {}
    has_changed_container = any(
        state["source_items"] != state["original_source_items"] for state in ordered_states
    )
    for state in ordered_states:
        if state["source_items"] == state["original_source_items"]:
            if has_changed_container and state["initializations"]:
                raise UserError(
                    "SOURCE_PLAN_TRANSIENT_FACTORY_UNSUPPORTED",
                    "计划包含最终未改变的容器，但其临时 Layer 工厂可能影响其他修改。",
                    "源码没有写入；候选工厂可能已经执行且副作用不会回滚。拆分计划或直接编辑源码后重验。",
                )
            continue
        # Every factory ran during preview. Even an intermediate layer that
        # is later removed can affect later construction through user code or
        # mutable external state; omitting it would change saved behavior.
        initializations = state["initializations"]
        _, offset, snippet = runtime.source_sequential_rebind(
            state["source_bytes"], state["tree"], state["statement"],
            state["assignment_target"], state["source_items"],
            initializations, namespace=state["namespace"], return_patch=True,
            container_type=state["container_type"],
            stages=state["stages"], original_alias=state["original_alias"],
            original_items=state["original_source_items"],
        )
        patches_by_file.setdefault(state["relative"].as_posix(), []).append((state, offset, snippet))

    if not patches_by_file:
        raise UserError(
            "SOURCE_NO_CHANGE",
            "结构计划最终没有改变任何容器。",
            "源码没有写入；候选模型和 Layer 工厂可能已经执行，其外部副作用不会回滚。",
        )

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
        **{key: value for key, value in runtime.inspect_model(model).items() if key in {"modules", "model_state_placement"}},
        "note": "成组 Sequential/ModuleList/ModuleDict CRUD 已统一构造、编译并按需前向检查；ModuleList 按索引重编号，ModuleDict 保留分支名与顺序。原容器构造代码仍会执行，被替换或删除的层仍可能先被构造；计划中随后移除的临时层工厂也会构造，可能产生副作用。预览在模型构造后应用计划，生成代码在各容器构造位置执行；跨容器或依赖中间状态的工厂执行顺序和可见状态不能保证一致，请检查差异并按需手工编辑。多次编辑会追加重建代码。提交使用可恢复事务日志与逐文件原子替换，重载/前向失败将整体回滚。文件系统不提供跨文件瞬时原子可见性。",
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
    for name, node in static_keyword_nodes(call).items():
        values[name] = expression_value(node, name)

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


def source_set_transaction_candidate(runtime, request, edit, *, shared_model=None):
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
    model, origin, _, _ = shared_model or runtime.load_model({**request, "variant": None})
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
        if not is_project_python_source(root, resolved):
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
                source_bytes = read_bounded_python_source(resolved)
                source_bytes_total += len(source_bytes)
                if source_bytes_total > 64 * 1024 * 1024:
                    raise UserError("SOURCE_PLAN_TOO_LARGE", "源码原子计划的原文件总量超过 64 MiB。", "减少计划涉及的文件数。")
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
                f"{source_name}:{site['line']} 找到 {len(calls)} 个 {type(old).__name__} 构造调用，无法可靠绑定 {target_path}。",
                ("该层可能由别名或工厂间接构造；请检查调用方并手动编辑源码，或使用变体。"
                 if not calls else "将目标 Layer 构造移到独立代码行后重试，或使用变体。"),
                {"source_edit_location": {"source": source_name, "line": site["line"],
                 "digest": hashlib.sha256(source_info["bytes"]).hexdigest(), "path": target_path,
                 "identity_scope": "source-read-during-preview"}},
            )
        call = calls[0]
        keyword_nodes = static_keyword_nodes(call)
        insertion_values = []
        for name, value in parameters.items():
            if name in keyword_nodes:
                node = keyword_nodes[name]
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
        **{key: value for key, value in runtime.inspect_model(model).items() if key in {"modules", "model_state_placement"}},
        "note": "多文件 set 计划已统一构造和检查；提交使用可恢复事务日志与逐文件原子替换，任一重载/前向失败将整体回滚。文件系统不提供跨文件瞬时原子可见性。",
    }


def source_mixed_transaction_candidate(runtime, request, edit):
    """Combine constructor edits and container edits before the one forward.

    Paths are interpreted against the original model: set operations must
    precede structural operations. Overlapping source hunks and ancestor edits
    fail closed rather than guessing the user's intended operation order.
    """
    operations = edit.get("operations")
    if not isinstance(operations, list) or not 2 <= len(operations) <= 64:
        raise UserError("SOURCE_EDIT_UNSUPPORTED", "混合计划需要 2..64 项源码操作。")
    sets, structures, structural_started = [], [], False
    for operation in operations:
        if not isinstance(operation, dict):
            raise UserError("SOURCE_EDIT_UNSUPPORTED", "混合计划中的每项操作必须是对象。")
        action = operation.get("action")
        if action == "set" and not structural_started:
            sets.append(operation)
        elif action in {"add", "replace", "remove"}:
            structural_started = True
            structures.append(operation)
        else:
            raise UserError("SOURCE_PLAN_ORDER_UNSUPPORTED", "混合源码计划要求先列出所有 set，再列出结构操作。",
                            "调整操作顺序；不会修改源码。")
    if not sets or not structures:
        raise UserError("SOURCE_EDIT_UNSUPPORTED", "混合计划需要参数修改和结构操作。")
    set_paths = [runtime.normalize_path(item.get("path")) for item in sets]
    for item in structures:
        path = runtime.normalize_path(item.get("parent") if item["action"] == "add" else item.get("path"))
        for target in set_paths:
            if path == target or path.startswith(target + ".") or (item["action"] != "add" and target.startswith(path + ".")):
                raise UserError("SOURCE_PLAN_OVERLAPPING_TARGETS", f"参数目标 {target} 与结构目标 {path} 重叠。",
                                "将重叠修改写到源码，或拆分成明确的步骤。")
    shared = runtime.load_model({**request, "variant": None})
    no_forward = {**request, "batch": None, "require_forward": False}
    parameter = source_set_transaction_candidate(runtime, no_forward, {"operations": sets}, shared_model=shared)
    structure = source_sequential_transaction_candidate(runtime, request, {"operations": structures}, shared_model=shared)
    root = Path(shared[1]["project_root"]).resolve()
    grouped = {}
    for part in (parameter, structure):
        for patch in part["source_patches"]:
            grouped.setdefault(patch["path"], []).append(patch)
    merged = []
    total_bytes = 0
    for name, patches in grouped.items():
        path = root / name
        try:
            if not path.is_file() or path.resolve(strict=True) != path or path.stat().st_size > 16 * 1024 * 1024:
                raise UserError("SOURCE_PATH_UNSUPPORTED", f"合并时源码不可用或路径经过符号链接：{name}。")
            original = read_bounded_python_source(path)
        except OSError as error:
            raise UserError("SOURCE_PLAN_CONFLICT", f"无法重新读取计划源码 {name}：{error}") from error
        original_digest = hashlib.sha256(original).hexdigest()
        if any(patch["before_sha256"] != original_digest for patch in patches):
            raise UserError("SOURCE_PLAN_CONFLICT", f"合并候选时源码已变化：{name}。", "重新预览；源码未修改。")
        lines = original.splitlines(keepends=True)
        if len(lines) > 30000:
            raise UserError("SOURCE_PLAN_TOO_LARGE", f"混合计划涉及的文件行数过多：{name}。", "拆分计划或手工编辑源码。")
        changes = []
        for patch in patches:
            candidate = patch["contents"].encode("utf-8").splitlines(keepends=True)
            if len(candidate) > 30000:
                raise UserError("SOURCE_PLAN_TOO_LARGE", f"混合候选行数过多：{name}。")
            # Large generated files commonly repeat identical declarations;
            # avoid quadratic matching on frequent lines. Coarse hunks may
            # conservatively conflict, but are never silently overwritten.
            matcher = difflib.SequenceMatcher(None, lines, candidate, autojunk=True)
            for tag, start, end, other_start, other_end in matcher.get_opcodes():
                if tag != "equal":
                    changes.append((start, end, b"".join(candidate[other_start:other_end])))
        changes.sort(key=lambda item: (item[0], item[1]))
        for previous, current in zip(changes, changes[1:]):
            # An insertion immediately before a replacement is unambiguous:
            # reverse application replaces the original line first, then
            # inserts before it. Two insertions at the same point have no
            # defined order and must still be rejected.
            if current[0] < previous[1] or (current[0] == previous[0] and current[0] == current[1] and previous[0] == previous[1]):
                raise UserError("SOURCE_PLAN_OVERLAP", f"混合计划在 {name} 的同一源码行或插入位置发生重叠。",
                                "调整源码或拆分计划；没有修改任何文件。")
        candidate_lines = list(lines)
        for start, end, replacement in reversed(changes):
            candidate_lines[start:end] = [replacement]
        candidate_bytes = b"".join(candidate_lines)
        try:
            original_text = original.decode("utf-8")
            candidate_text = candidate_bytes.decode("utf-8")
            compile(candidate_text, str(path), "exec")
        except (UnicodeDecodeError, SyntaxError) as error:
            raise UserError("SOURCE_PATCH_INVALID", f"混合源码候选无法解析：{name}: {error}") from error
        diff = "".join(difflib.unified_diff(original_text.splitlines(keepends=True), candidate_text.splitlines(keepends=True),
                                            fromfile=name, tofile=name + " (candidate)"))
        total_bytes += len(candidate_bytes) + 2 * len(diff.encode("utf-8"))
        if total_bytes > 24 * 1024 * 1024:
            raise UserError("SOURCE_PLAN_TOO_LARGE", "混合计划预览超过 24 MiB。")
        merged.append({"path": name, "line": min((patch.get("line") or 1) for patch in patches),
                       "operations": [item for patch in patches for item in patch.get("operations", [])],
                       "before_sha256": original_digest, "after_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                       "unified_diff": diff, "contents": candidate_text})
    primary = dict(merged[0])
    primary["unified_diff"] = "".join(patch["unified_diff"] for patch in merged)
    primary.pop("contents", None)
    if len(merged) > 1:
        primary["path"], primary["line"] = f"{len(merged)} files", None
        primary["operations"] = [{"path": patch["path"], "operations": patch["operations"]} for patch in merged]
    structure["source_patch"], structure["source_patches"] = primary, merged
    structure["edit_effects"] = parameter["edit_effects"] + structure["edit_effects"]
    structure["note"] = "参数修改先于结构修改作用于同一临时模型；合并源码后提交仍须在新 worker 中重载并验证。多文件不保证瞬时原子可见。"
    return structure
