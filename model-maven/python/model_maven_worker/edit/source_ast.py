"""Readable Python AST helpers for source-level Layer edits."""

import ast
import json
import re

from ..protocol import UserError


def ast_offset(lines, line, column):
    return sum(len(part) for part in lines[: line - 1]) + column


def source_argument_insertions(source_bytes, call, values):
    """Insert omitted explicit keywords without discarding multiline comments."""
    lines = source_bytes.splitlines(keepends=True)
    close = ast_offset(lines, call.end_lineno, call.end_col_offset) - 1
    if source_bytes[close:close + 1] != b")":
        raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "无法识别构造调用的结束位置。")
    encoded = [f"{name}={value!r}".encode("utf-8") for name, value in values]
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
    arguments = ", ".join(f"{name}={value!r}" for name, value in parameters.items())
    return f"{torch_alias}.nn.{layer_type}({arguments})"


def source_factory_expression(component, parameters, tree, operation_index):
    if not isinstance(component, str) or component.count(":") != 1:
        raise UserError("ENTRYPOINT_INVALID", "自定义 Layer 工厂入口应为 module:callable。")
    module_name, attribute = component.split(":")
    if not all(part.isidentifier() for part in module_name.split(".")) or not all(
        part.isidentifier() for part in attribute.split(".")
    ):
        raise UserError("ENTRYPOINT_INVALID", "自定义 Layer 工厂入口只能包含模块/属性名称。")
    module_alias = unique_source_alias(tree, f"_m_model_component_{operation_index}")
    json_alias = unique_source_alias(tree, f"_m_model_json_{operation_index}")
    try:
        encoded = json.dumps(
            parameters, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise UserError("FIELD_INVALID", f"自定义 Layer 参数不是有限 JSON 值：{error}") from error
    expression = (
        f"{module_alias}.{attribute}(**{json_alias}.loads({encoded!r}))"
    )
    imports = [
        f"import {module_name} as {module_alias}",
        f"import json as {json_alias}",
    ]
    return expression, imports


def source_sequential_rebind(
    source_bytes, tree, statement, target, items, initialization=None,
    namespace=0, return_patch=False, container_type="Sequential",
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
            "Sequential 必须直接赋给局部变量或对象属性。",
            "先将容器单独赋值给变量或对象属性，再执行源码 CRUD。",
        )
    lines = source_bytes.splitlines(keepends=True)
    if statement.end_lineno is None or statement.end_lineno > len(lines):
        raise UserError("SOURCE_CALLSITE_UNSUPPORTED", "无法确定 Sequential 赋值的结束位置。")
    line = lines[statement.end_lineno - 1]
    content = line.rstrip(b"\r\n")
    indent = content[:statement.col_offset]
    suffix = content[statement.end_col_offset:].strip()
    if indent.strip() or (suffix and not suffix.startswith(b"#")):
        raise UserError(
            "SOURCE_CALLSITE_UNSUPPORTED",
            "Sequential 赋值所在行还包含其他语句，不能安全插入重建代码。",
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
    if container_type == "Sequential":
        snippet_lines.append(f"from collections import OrderedDict as {ordered_alias}")
    if initialization is not None:
        initializations = initialization if isinstance(initialization, list) else [initialization]
        imported = set()
        for item in initializations:
            for import_line in item.get("imports", []):
                if import_line not in imported:
                    snippet_lines.append(import_line)
                    imported.add(import_line)
            if item.get("component_note"):
                snippet_lines.append(f"# {item['component_note']}")
            snippet_lines.extend([
                f"with {torch_alias}.random.fork_rng(devices=[]):",
                f"    {torch_alias}.random.default_generator.manual_seed({item['seed']})",
                f"    {item['name']} = {item['expression']}",
            ])
    if container_type == "ModuleList":
        snippet_lines.append(f"{target_expression} = {torch_alias}.nn.ModuleList([")
        snippet_lines.extend(f"    {expression}," for _, expression in items)
        snippet_lines.append("])")
    elif container_type == "ModuleDict":
        snippet_lines.append(f"{target_expression} = {torch_alias}.nn.ModuleDict({{")
        snippet_lines.extend(f"    {name!r}: {expression}," for name, expression in items)
        snippet_lines.append("})")
    else:
        snippet_lines.extend([
            f"{target_expression} = {torch_alias}.nn.Sequential(",
            f"    {ordered_alias}([",
        ])
        snippet_lines.extend(f"        ({name!r}, {expression})," for name, expression in items)
        snippet_lines.extend(["    ])", ")"])
    snippet = newline.join(indent + item.encode("utf-8") for item in snippet_lines) + newline
    insertion_line = statement.end_lineno
    marker = f"# Rebuild this {container_type} after the explicit Model-Maven Layer edit.".encode()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.end_lineno is None or node.lineno <= statement.end_lineno:
            continue
        if not any(ast.unparse(candidate) == target_expression for candidate in targets):
            continue
        markers = [
            index + 1 for index, source_line in enumerate(lines)
            if marker in source_line
            and statement.end_lineno < index + 1 < node.lineno
        ]
        if markers:
            insertion_line = max(insertion_line, node.end_lineno)
    insertion = sum(len(part) for part in lines[:insertion_line])
    insertion_line_content = lines[insertion_line - 1]
    if not insertion_line_content.endswith((b"\n", b"\r")):
        snippet = newline + snippet
    candidate = source_bytes[:insertion] + snippet + source_bytes[insertion:]
    if return_patch:
        return candidate, insertion, snippet
    return candidate
