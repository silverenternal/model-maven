"""Candidate generation for setting explicit Layer constructor parameters."""

import ast
import difflib
import hashlib
import inspect
import os
from pathlib import Path

from ..protocol import UserError
from ..model.structure import shared_state_groups
from .source_ast import source_argument_insertions


def source_set_candidate(runtime, request, edit):
    if request.get("variant") is not None:
        raise UserError("SOURCE_VARIANT_CONFLICT", "源码编辑不能应用在已有结构变体上。")
    if edit.get("component_ref") is not None or edit.get("component") is not None:
        raise UserError("SOURCE_EDIT_UNSUPPORTED", "源码字段编辑暂不支持通过组件替换工厂目标。")
    model, origin, _, _ = runtime.load_model({**request, "variant": None})
    target_path = runtime.normalize_path(edit.get("path"))
    old = runtime.resolve_module(model, target_path)
    source_locations = runtime.source_locations()
    source_site_counts = runtime.source_site_counts()
    site = source_locations.get(id(old))
    if not site:
        raise UserError(
            "SOURCE_LOCATION_UNAVAILABLE",
            f"无法定位 {target_path} 的项目构造代码。",
            "该 Layer 可能由动态工厂、外部包或未追踪控制流创建；源码保持不变。",
        )
    same_site_instances = [
        module
        for module in runtime.module_entries(model)[0].values()
        if type(module) is type(old)
        and (candidate := source_locations.get(id(module))) is not None
        and candidate.get("source_path") == site.get("source_path")
        and candidate.get("line") == site.get("line")
    ]
    site_key = (type(old).__module__, type(old).__qualname__, site.get("source_path"), site.get("line"))
    if len(same_site_instances) > 1 or source_site_counts.get(site_key, 0) > 1:
        raise UserError(
            "SOURCE_CALLSITE_SHARED",
            f"第 {site['line']} 行构造了多个 {type(old).__name__} 实例，无法把模块路径唯一对应到单个源码参数。",
            "把目标 Layer 的构造拆到独立代码行，或使用 --target variant。",
        )
    source_path = Path(site.get("source_path", site.get("file", "")))
    root = Path(origin["project_root"]).resolve()
    try:
        lexical_path = Path(os.path.abspath(source_path))
        relative = lexical_path.relative_to(root)
        resolved = lexical_path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise UserError("SOURCE_PATH_UNSUPPORTED", "Layer 构造文件不在当前项目目录内。", "只允许编辑项目内、可写的 Python 源码。") from error
    ignored = {".venv", "venv", "site-packages", "target", ".git", "__pycache__", ".model-maven"}
    if resolved.suffix != ".py" or any(part in ignored for part in relative.parts):
        raise UserError("SOURCE_PATH_UNSUPPORTED", "Layer 构造文件不属于可编辑的项目 Python 源码。")
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

    parameters = edit.get("params", {})
    fields = runtime.source_editable_fields(old)
    if not isinstance(parameters, dict) or not parameters or set(parameters) - fields:
        raise UserError(
            "FIELD_INVALID",
            f"该 Layer 当前可通过源码参数处理器修改：{', '.join(sorted(fields)) or '无可识别构造参数'}",
        )
    signature = inspect.signature(type(old).__init__)
    parameter_names = [
        name for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    candidates = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.lineno == site["line"]
        and runtime.call_name(node) == type(old).__name__
    ]
    if len(candidates) != 1:
        raise UserError(
            "SOURCE_CALLSITE_AMBIGUOUS",
            f"第 {site['line']} 行找到 {len(candidates)} 个 {type(old).__name__} 构造调用，无法可靠绑定目标。",
            "把目标 Layer 构造移到单独一行，或继续使用 --target variant。",
        )
    call = candidates[0]
    if any(isinstance(argument, ast.Starred) for argument in call.args) or any(keyword.arg is None for keyword in call.keywords):
        raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "暂不改写 *args 或 **kwargs 构造表达式。", "改用显式参数名，或使用 --target variant。")

    byte_lines = source_bytes.splitlines(keepends=True)
    replacements = []
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
        start = runtime.ast_offset(byte_lines, node.lineno, node.col_offset)
        end = runtime.ast_offset(byte_lines, node.end_lineno, node.end_col_offset)
        replacements.append((start, end, repr(value).encode("utf-8")))

    if insertion_values:
        replacements.extend(source_argument_insertions(source_bytes, call, insertion_values))

    candidate_bytes = source_bytes
    for start, end, replacement in sorted(replacements, reverse=True):
        candidate_bytes = candidate_bytes[:start] + replacement + candidate_bytes[end:]
    try:
        candidate_source = candidate_bytes.decode("utf-8")
        compile(candidate_source, str(resolved), "exec")
    except (UnicodeDecodeError, SyntaxError) as error:
        raise UserError("SOURCE_PATCH_INVALID", f"源码补丁未通过语法检查：{error}", "原文件保持不变。") from error

    operation = {key: value for key, value in edit.items() if key not in {"target", "component_ref"}}
    effects = runtime.apply_source_set_candidate(model, operation, call)
    if "preserved_external_state_sharing" in effects:
        effects["candidate_external_state_sharing"] = effects.pop("preserved_external_state_sharing")
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
            "path": relative_name,
            "line": site["line"],
            "layer": target_path,
            "before_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "after_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "unified_diff": diff,
            "contents": candidate_source,
        },
        "validation": validation,
        "shared_state_evidence": {"scope": "in_memory_candidate_object_storage_v2", "groups": shared_state_groups(model)},
        "edit_effects": [effects],
        "modules": runtime.inspect_model(model)["modules"],
        "note": "源码目标已通过内存候选与语法检查；提交时将原子替换单个 .py 文件，之后以新 worker 重建并按需真实前向，失败会按 SHA-256 回滚。",
    }
