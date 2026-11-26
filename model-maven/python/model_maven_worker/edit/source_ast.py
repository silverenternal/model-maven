"""Readable Python AST helpers for source-level Layer edits."""

import ast
import json
import keyword
import re
import unicodedata

from ..protocol import UserError


def read_bounded_python_source(path):
    """Read a candidate source without trusting a prior stat size check."""
    limit = 16 * 1024 * 1024
    with path.open("rb") as stream:
        source = stream.read(limit + 1)
    if len(source) > limit:
        raise UserError("SOURCE_TOO_LARGE", "源码文件超过 16 MiB，暂不支持自动编辑。")
    return source


def ast_offset(lines, line, column):
    return sum(len(part) for part in lines[: line - 1]) + column


def static_keyword_nodes(call):
    """Map explicit keywords and literal **dict entries to their value nodes."""
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "暂不改写包含 *args 的构造表达式。")
    result = {}
    for item in call.keywords:
        if item.arg is not None:
            pairs = [(item.arg, item.value)]
        elif isinstance(item.value, ast.Dict):
            pairs = []
            for key, value in zip(item.value.keys, item.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "仅支持键为字符串字面量的 ** 字典。")
                pairs.append((key.value, value))
        else:
            raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "暂不改写动态 **kwargs 构造表达式。")
        for name, value in pairs:
            if name in result:
                raise UserError("SOURCE_CALLSITE_UNSUPPORTED", f"构造参数 {name!r} 存在重复声明。")
            result[name] = value
    return result


def source_argument_insertions(source_bytes, call, values):
    """Insert omitted explicit keywords without discarding multiline comments."""
    lines = source_bytes.splitlines(keepends=True)
    close = ast_offset(lines, call.end_lineno, call.end_col_offset) - 1
    if source_bytes[close:close + 1] != b")":
        raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "无法识别构造调用的结束位置。")
    encoded = []
    for name, value in values:
        if (isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)
                and unicodedata.normalize("NFKC", name) == name):
            encoded.append(f"{name}={value!r}".encode("utf-8"))
        else:
            encoded.append(f"**{{{name!r}: {value!r}}}".encode("utf-8"))
    arguments = list(call.args) + [keyword.value for keyword in call.keywords]
    last = max(arguments, key=lambda n: (n.end_lineno, n.end_col_offset), default=None)
    last_end = ast_offset(lines, last.end_lineno, last.end_col_offset) if last else close
    # Outside the last expression only grouping delimiters, separators,
    # whitespace, line continuations and comments can precede the final ')'.
    significant = None
    significant_end = last_end
    comment = False
    for offset in range(last_end, close):
        byte = source_bytes[offset]
        if byte in (10, 13):
            comment = False
        elif comment:
            continue
        elif byte == 35:  # #
            comment = True
        elif byte not in b" \t\f\\":
            significant = byte
            significant_end = offset + 1
    has_comma = significant == 44
    line_start = ast_offset(lines, call.end_lineno, 0)
    closing_indent = source_bytes[line_start:close]
    if call.end_lineno != call.lineno and not closing_indent.strip():
        newline = b"\r\n" if any(line.endswith(b"\r\n") for line in lines[call.lineno - 1:call.end_lineno]) else b"\n"
        indent = closing_indent + b"    "
        if last is not None:
            last_line = lines[last.lineno - 1]
            last_indent = last_line[:len(last_line) - len(last_line.lstrip(b" \t"))]
            if len(last_indent) > len(closing_indent):
                indent = last_indent
        edits = []
        if last is not None and not has_comma:
            edits.append((significant_end, significant_end, b","))
        insertion = b"".join(indent + item + b"," + newline for item in encoded)
        edits.append((line_start, line_start, insertion))
        return edits
    separator = b" " if has_comma else (b", " if last else b"")
    return [(close, close, separator + b", ".join(encoded))]


def call_name(node):
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if (isinstance(target, ast.Call)
            and isinstance(target.func, ast.Name) and target.func.id == "getattr"
            and len(target.args) == 2 and not target.keywords
            and isinstance(target.args[0], ast.Attribute) and target.args[0].attr == "nn"
            and isinstance(target.args[1], ast.Constant) and isinstance(target.args[1].value, str)):
        return target.args[1].value
    return None


def unique_source_alias(tree, base):
    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", ast.unparse(tree)))
    candidate = base
    suffix = 1
    while candidate in identifiers:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def source_module_expression(layer_type, parameters, torch_alias):
    # Python normalizes identifiers in source, while **kwargs preserves the
    # supplied string keys. Avoid changing the factory call for unusual names.
    if all(isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)
           and unicodedata.normalize("NFKC", name) == name for name in parameters):
        arguments = ", ".join(f"{name}={value!r}" for name, value in parameters.items())
    else:
        arguments = f"**{parameters!r}"
    if layer_type.isascii() and layer_type.isidentifier() and not keyword.iskeyword(layer_type):
        return f"{torch_alias}.nn.{layer_type}({arguments})"
    return f"getattr({torch_alias}.nn, {layer_type!r})({arguments})"


def source_factory_expression(component, parameters, tree, operation_index):
    if not isinstance(component, str) or component.count(":") != 1:
        raise UserError("ENTRYPOINT_INVALID", "自定义 Layer 工厂入口应为 module:callable。")
    module_name, attribute = component.split(":")
    if not all(part.isidentifier() for part in module_name.split(".")) or not all(
        part.isidentifier() for part in attribute.split(".")
    ):
        raise UserError("ENTRYPOINT_INVALID", "自定义 Layer 工厂入口只能包含模块/属性名称。")
    module_alias = unique_source_alias(tree, f"_m_model_component_{operation_index}")
    try:
        json.dumps(
            parameters, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise UserError("FIELD_INVALID", f"自定义 Layer 参数不是有限 JSON 值：{error}") from error
    parts = module_name.split(".") + attribute.split(".")
    safe_syntax = all(not keyword.iskeyword(part) and unicodedata.normalize("NFKC", part) == part
                      for part in parts)
    if safe_syntax:
        expression = f"{module_alias}.{attribute}(**{parameters!r})"
        imports = [f"import {module_name} as {module_alias}"]
    else:
        importlib_alias = unique_source_alias(tree, f"_m_model_importlib_{operation_index}")
        builtins_alias = unique_source_alias(tree, f"_m_model_builtins_{operation_index}")
        target = module_alias
        for part in attribute.split("."):
            target = f"{builtins_alias}.getattr({target}, {part!r})"
        expression = f"{target}(**{parameters!r})"
        imports = [f"import importlib as {importlib_alias}", f"import builtins as {builtins_alias}",
                   f"{module_alias} = {importlib_alias}.import_module({module_name!r})"]
    return expression, imports


def source_sequential_rebind(
    source_bytes, tree, statement, target, items, initialization=None,
    namespace=0, return_patch=False, container_type="Sequential", stages=None,
    original_alias=None, original_items=None,
):
    def simple_target(node):
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Attribute):
            return simple_target(node.value)
        return False

    if not simple_target(target):
        raise UserError(
            "SOURCE_CALLSITE_UNSUPPORTED",
            f"{container_type} 必须直接赋给局部变量或对象属性。",
            "先将容器单独赋值给变量或对象属性，再执行源码 CRUD。",
        )
    lines = source_bytes.splitlines(keepends=True)
    if statement.end_lineno is None or statement.end_lineno > len(lines):
        raise UserError("SOURCE_CALLSITE_UNSUPPORTED", f"无法确定 {container_type} 赋值的结束位置。")
    line = lines[statement.end_lineno - 1]
    content = line.rstrip(b"\r\n")
    indent = content[:statement.col_offset]
    suffix = content[statement.end_col_offset:].strip()
    if indent.strip() or (suffix and not suffix.startswith(b"#")):
        raise UserError(
            "SOURCE_CALLSITE_UNSUPPORTED",
            f"{container_type} 赋值所在行还包含其他语句，不能安全插入重建代码。",
            "把容器赋值单独放在一行后重试。",
        )
    newline = b"\r\n" if line.endswith(b"\r\n") else b"\n"
    suffix = f"_{namespace}" if namespace else ""
    torch_alias = unique_source_alias(tree, f"_m_model_torch{suffix}")
    ordered_alias = unique_source_alias(tree, f"_m_model_ordered_dict{suffix}")
    target_expression = ast.unparse(target)
    snippet_lines = [
        f"# Rebuild this {container_type} after the explicit Model-Maven Layer edit.",
        f"import torch as {torch_alias}",
    ]
    if container_type == "Sequential" or stages is not None:
        snippet_lines.append(f"from collections import OrderedDict as {ordered_alias}")
    imported = set()

    def emit_initialization(item):
        if item is not None:
            for import_line in item.get("imports", []):
                if import_line not in imported:
                    snippet_lines.append(import_line)
                    imported.add(import_line)
            if item.get("component_note"):
                snippet_lines.append(f"# {item['component_note']}")
            if item.get("seed") is None:
                snippet_lines.append(f"{item['name']} = {item['expression']}")
            else:
                snippet_lines.extend([
                    f"with {torch_alias}.random.fork_rng(devices=[]):",
                    f"    {torch_alias}.random.default_generator.manual_seed({item['seed']})",
                    f"    {item['name']} = {item['expression']}",
                ])
            if item.get("training_from"):
                snippet_lines.append(f"{item['name']}.train(({item['training_from']}).training)")
    if stages is not None:
        staged_alias = unique_source_alias(tree, f"_m_model_staged_modules{suffix}")
        snippet_lines.append(f"{original_alias} = {target_expression}._modules.copy()")
        snippet_lines.append(f"{staged_alias} = {ordered_alias}({original_alias})")
        previous = list(original_items or [])
        for stage in stages:
            emit_initialization(stage.get("initialization"))
            current = stage["items"]
            previous_names = [name for name, _ in previous]
            current_names = [name for name, _ in current]
            previous_map = dict(previous)
            current_map = dict(current)
            updated = [name for name in current_names if name in previous_map and previous_map[name] != current_map[name]]
            removed = [name for name in previous_names if name not in current_map]
            added = [name for name in current_names if name not in previous_map]
            if previous_names == current_names and len(updated) == 1:
                name = updated[0]
                snippet_lines.append(f"{staged_alias}[{name!r}] = {current_map[name]}")
            elif not updated and not removed and len(added) == 1 and current_names == previous_names + added:
                name = added[0]
                snippet_lines.append(f"{staged_alias}[{name!r}] = {current_map[name]}")
            elif not updated and not added and len(removed) == 1 and current_names == [name for name in previous_names if name != removed[0]]:
                snippet_lines.append(f"del {staged_alias}[{removed[0]!r}]")
            elif previous != current:
                # Insertion, reindexing, or more complex changes need the
                # explicit order; do not guess from a partial delta.
                snippet_lines.append(f"{staged_alias} = {ordered_alias}([")
                snippet_lines.extend(f"    ({name!r}, {expression})," for name, expression in current)
                snippet_lines.append("])")
            snippet_lines.extend([
                f"{target_expression}._modules.clear()",
                f"{target_expression}._modules.update({staged_alias})",
            ])
            previous = list(current)
    elif initialization is not None:
        initializations = initialization if isinstance(initialization, list) else [initialization]
        for item in initializations:
            emit_initialization(item)
    if stages is not None:
        # The final staged mapping already contains the exact ordered modules.
        # Reuse it instead of emitting and evaluating the whole list again.
        if not stages:
            raise UserError("SOURCE_PLAN_STATE_INVALID", "源码结构计划缺少已执行的操作。")
        if container_type == "ModuleList":
            argument = f"[*{staged_alias}.values()]"
        elif container_type == "ModuleDict":
            argument = f"{{**{staged_alias}}}"
        else:
            argument = staged_alias
        snippet_lines.append(f"{target_expression} = {torch_alias}.nn.{container_type}({argument})")
    else:
        # A fixed ordered key list plus at most one different expression can
        # represent append, insert, replace, or remove without repeating the
        # target expression for every child. ModuleList source indices are
        # retained separately from their renumbered destination positions.
        compact = len(items) > 8 and all(isinstance(name, str) for name, _ in items)
        if container_type == "ModuleList":
            compact = compact and all(name.isdecimal() for name, _ in items)
        special = []
        source_keys = []
        if compact:
            for name, expression in items:
                if container_type == "ModuleList":
                    match = re.fullmatch(re.escape(target_expression) + r"\[([0-9]+)\]", expression)
                    if match:
                        source_keys.append(int(match.group(1)))
                    else:
                        source_keys.append(None)
                        special.append((None, expression))
                else:
                    source_keys.append(name)
                    expected = (f"{target_expression}[{name!r}]" if container_type == "ModuleDict"
                                else f"{target_expression}._modules[{name!r}]")
                    if expression != expected:
                        special.append((name, expression))
            compact = len(special) <= 1
    if stages is None and compact:
        pairs_alias = unique_source_alias(tree, f"_m_model_children{suffix}")
        key_alias = unique_source_alias(tree, f"_m_model_child_key{suffix}")
        keys = source_keys
        lookup = (f"{target_expression}[{key_alias}]" if container_type in {"ModuleList", "ModuleDict"}
                  else f"{target_expression}._modules[{key_alias}]")
        snippet_lines.append(f"{pairs_alias} = [")
        if special:
            changed_key, changed_expression = special[0]
            if container_type != "ModuleList":
                snippet_lines.extend(["    (", f"        {key_alias},", "        ("])
                expression_indent = "            "
            else:
                snippet_lines.append("    (")
                expression_indent = "        "
            condition = f"{key_alias} is None" if changed_key is None else f"{key_alias} == {changed_key!r}"
            snippet_lines.extend([
                f"{expression_indent}{changed_expression} if {condition}",
                f"{expression_indent}else {lookup}",
            ])
            snippet_lines.extend(["        ),", "    )"] if container_type != "ModuleList" else ["    )"])
        else:
            expression = lookup if container_type == "ModuleList" else f"({key_alias}, {lookup})"
            snippet_lines.append(f"    {expression}")
        snippet_lines.append(f"    for {key_alias} in (")
        for start in range(0, len(keys), 6):
            snippet_lines.append("        " + ", ".join(repr(key) for key in keys[start:start + 6]) + ",")
        snippet_lines.extend(["    )", "]"])
        if container_type == "ModuleList":
            argument = pairs_alias
        elif container_type == "ModuleDict":
            argument = f"{{key: value for key, value in {pairs_alias}}}"
        else:
            argument = f"{ordered_alias}({pairs_alias})"
        snippet_lines.append(f"{target_expression} = {torch_alias}.nn.{container_type}({argument})")
    elif stages is None and container_type == "ModuleList":
        snippet_lines.append(f"{target_expression} = {torch_alias}.nn.ModuleList([")
        snippet_lines.extend(f"    {expression}," for _, expression in items)
        snippet_lines.append("])")
    elif stages is None and container_type == "ModuleDict":
        snippet_lines.append(f"{target_expression} = {torch_alias}.nn.ModuleDict({{")
        snippet_lines.extend(f"    {name!r}: {expression}," for name, expression in items)
        snippet_lines.append("})")
    elif stages is None:
        snippet_lines.extend([
            f"{target_expression} = {torch_alias}.nn.Sequential(",
            f"    {ordered_alias}([",
        ])
        snippet_lines.extend(f"        ({name!r}, {expression})," for name, expression in items)
        snippet_lines.extend(["    ])", ")"])
    snippet = newline.join(indent + item.encode("utf-8") for item in snippet_lines) + newline
    # This statement is the construction site observed for the selected
    # container. Later assignments can refer to a different runtime object,
    # even when they have the same target and a generated-looking comment.
    insertion_line = statement.end_lineno
    insertion = sum(len(part) for part in lines[:insertion_line])
    insertion_line_content = lines[insertion_line - 1]
    if not insertion_line_content.endswith((b"\n", b"\r")):
        snippet = newline + snippet
    candidate = source_bytes[:insertion] + snippet + source_bytes[insertion:]
    if return_patch:
        return candidate, insertion, snippet
    return candidate
