"""Conservative, source-preserving migration of early generated starters.

This module only produces a candidate; publication and runtime validation must
be performed by the source-edit transaction that invokes it.
"""

import ast
import difflib
import hashlib
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import random
import sys

from ..protocol import UserError
from ..model.structure import shared_state_groups


def _unsupported(reason):
    raise UserError(
        "STARTER_MIGRATION_UNSUPPORTED",
        reason,
        "保留原文件；请手工适配工厂和 forward，或在新目录创建新版骨架并迁移自定义代码。",
    )


def migrate_starter_source(source):
    """Return a candidate string for a recognizable first-generation starter.

    Only the two signatures, model call, generated encoder expressions, and
    new declarations/helpers are edited. Existing comments and other code are
    preserved. The caller must independently compare and verify the candidate.
    """
    if not isinstance(source, str) or not source.startswith("# Editable Model-Maven starter."):
        _unsupported("文件不是可识别的 Model-Maven 旧骨架。")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        _unsupported(f"无法解析骨架源码：{error}")
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"]
    factories = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "model"]
    if len(classes) != 1 or len(factories) != 1:
        _unsupported("需要唯一的 Model 类和 model 工厂。")
    cls, factory = classes[0], factories[0]
    init = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    forward = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    if len(init) != 1 or len(forward) != 1 or cls.col_offset != 0:
        _unsupported("需要唯一的标准 Model.__init__ 和 forward。")
    init, forward = init[0], forward[0]
    if forward.decorator_list:
        _unsupported("forward 使用了装饰器，无法确认迁移后语义。")
    if init.col_offset != 4 or forward.col_offset != 4 or factory.col_offset != 0:
        _unsupported("骨架缩进与受支持的生成格式不同。")
    if any((isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "_m_model_encode")
           or (isinstance(n, ast.Name) and n.id == "_m_model_encode")
           or (isinstance(n, ast.Attribute) and n.attr == "_m_model_encode") for n in ast.walk(cls)):
        _unsupported("模型已包含 _m_model_encode，不能覆盖用户方法。")
    if any((isinstance(n, ast.Name) and n.id in ("input_routes", "encoder_output_modes", "_m_model_inputs"))
           or (isinstance(n, ast.Attribute) and n.attr in ("input_routes", "native_encoder_outputs", "_model_maven_declared_routes"))
           for n in ast.walk(cls)):
        _unsupported("模型已包含新装配参数或同名变量，不能安全推断迁移方式。")
    if any(isinstance(n, ast.Name) and n.id in ("input_routes", "encoder_output_modes") for n in ast.walk(factory)):
        _unsupported("工厂已使用新装配参数的名称，不能安全引入同名局部变量。")
    for fn in (init, factory):
        args = fn.args
        if (fn.decorator_list or fn.returns is not None or args.posonlyargs or args.kwonlyargs
                or args.vararg or args.kwarg or any(arg.annotation is not None for arg in args.args)):
            _unsupported("工厂或构造器使用了装饰器或复杂参数，请手工适配。")
        if any(not isinstance(value, ast.Constant) or value.value is not None for value in args.defaults):
            _unsupported("工厂或构造器的参数默认值与旧骨架不同。")
    if any(n.arg in ("input_routes", "encoder_output_modes") for fn in (init, factory) for n in fn.args.args):
        _unsupported("工厂或构造器已声明新装配参数。")
    assignments = [n for n in init.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Attribute) and isinstance(n.targets[0].value, ast.Name)
                   and n.targets[0].value.id == "self" and n.targets[0].attr == "encoders"]
    values = [n for n in forward.body if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "values" and isinstance(n.value, ast.List)]
    returns = [n for n in factory.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == "Model"]
    if len(assignments) != 1 or len(values) != 1 or len(returns) != 1:
        _unsupported("找不到唯一的编码器赋值、输入列表和 Model 工厂调用。")
    if not (isinstance(assignments[0].value, ast.Call)
            and isinstance(assignments[0].value.func, ast.Attribute)
            and assignments[0].value.func.attr == "ModuleDict"):
        _unsupported("编码器赋值不是可识别的 ModuleDict 构造。")
    returns = returns[0].value
    if any(isinstance(argument, ast.Starred) for argument in returns.args) or any(keyword.arg is None or keyword.arg in ("input_routes", "encoder_output_modes") for keyword in returns.keywords):
        _unsupported("Model 工厂调用使用了动态或重复的装配参数。")
    encoded = []
    for entry in values[0].value.elts:
        if not (isinstance(entry, ast.Call) and isinstance(entry.func, ast.Attribute)
                and entry.func.attr == "flatten" and not entry.args and len(entry.keywords) == 1
                and entry.keywords[0].arg == "start_dim" and isinstance(entry.keywords[0].value, ast.Constant)
                and entry.keywords[0].value.value == 1):
            _unsupported("forward 的编码器输出不是可识别的展平形式。")
        call = entry.func.value
        if not (isinstance(call, ast.Call) and len(call.args) == 1 and not call.keywords
                and isinstance(call.func, ast.Subscript) and isinstance(call.func.value, ast.Attribute)
                and isinstance(call.func.value.value, ast.Name) and call.func.value.value.id == "self"
                and call.func.value.attr == "encoders" and isinstance(call.func.slice, ast.Constant)
                and isinstance(call.func.slice.value, str) and call.func.slice.value.startswith("enc_")
                and isinstance(call.args[0], ast.Subscript) and isinstance(call.args[0].value, ast.Name)
                and call.args[0].value.id == "inputs" and isinstance(call.args[0].slice, ast.Constant)
                and isinstance(call.args[0].slice.value, str)):
            _unsupported("forward 的编码器输入不是可识别的静态接线。")
        slot = call.func.slice.value[4:] + "_encoder"
        source_name = call.args[0].slice.value
        if not slot.isidentifier() or not source_name.isidentifier():
            _unsupported("编码器或输入名不适合自动迁移。")
        encoded.append((entry, slot, source_name))
    if not encoded or len({slot for _, slot, _ in encoded}) != len(encoded):
        _unsupported("编码器接线为空或重复。")

    raw = source.encode("utf-8")
    lines = raw.splitlines(keepends=True)
    def offset(line, column):
        return sum(map(len, lines[:line - 1])) + column
    def signature_close(fn):
        start = offset(fn.lineno, fn.col_offset)
        end = offset(fn.body[0].lineno, 0)
        header = raw[start:end]
        close = header.rfind(b")")
        if close < 0 or b"#" in header:
            _unsupported("无法安全修改工厂或构造器签名。")
        return start + close
    assignment_line = lines[assignments[0].end_lineno - 1].rstrip(b"\r\n")
    if assignment_line[assignments[0].end_col_offset:].strip() not in (b"",) and not assignment_line[assignments[0].end_col_offset:].lstrip().startswith(b"#"):
        _unsupported("编码器赋值行包含其他语句，不能安全插入迁移代码。")
    newline = "\r\n" if "\r\n" in source else "\n"
    edits = []
    for fn in (init, factory):
        point = signature_close(fn)
        edits.append((point, point, b", input_routes=None, encoder_output_modes=None"))
    end_call = offset(returns.end_lineno, returns.end_col_offset) - 1
    if raw[end_call:end_call + 1] != b")":
        _unsupported("无法修改 Model 工厂调用。")
    separator = b", " if returns.args or returns.keywords else b""
    edits.append((end_call, end_call, separator + b"input_routes=input_routes, encoder_output_modes=encoder_output_modes"))
    for entry, slot, _ in encoded:
        edits.append((offset(entry.lineno, entry.col_offset), offset(entry.end_lineno, entry.end_col_offset),
                      f"self._m_model_encode({slot!r}, inputs)".encode()))
    route_defaults = {slot: name for _, slot, name in encoded}
    input_names = sorted(set(route_defaults.values()))
    init_lines = [
        "        # Model-Maven: configurable starter routes and encoder output modes.",
        f"        self.input_routes = {route_defaults!r}",
        f"        _m_model_inputs = {input_names!r}",
        "        if input_routes is not None:",
        "            if not isinstance(input_routes, dict):",
        "                raise ValueError('input_routes must be a dictionary')",
        "            for slot, source in input_routes.items():",
        "                if slot not in self.input_routes or source not in _m_model_inputs:",
        "                    raise ValueError(f'Unknown route {slot}={source}')",
        "                self.input_routes[slot] = source",
        "        self.native_encoder_outputs = []",
        "        if encoder_output_modes is not None:",
        "            if not isinstance(encoder_output_modes, dict):",
        "                raise ValueError('encoder_output_modes must be a dictionary')",
        "            for slot, mode in encoder_output_modes.items():",
        "                if slot not in self.input_routes or mode not in ('native', 'flatten'):",
        "                    raise ValueError(f'Unknown encoder output mode {slot}={mode}')",
        "                if mode == 'native':",
        "                    self.native_encoder_outputs.append(slot)",
        "        self._model_maven_declared_routes = [",
        "            {'slot': slot, 'input': source, 'module_path': 'model.encoders.enc_' + slot[:-8]}",
        "            for slot, source in self.input_routes.items()",
        "        ]",
    ]
    point = offset(assignments[0].end_lineno + 1, 0)
    edits.append((point, point, (newline.join(init_lines) + newline).encode()))
    helper = [
        "    def _m_model_encode(self, slot, inputs):",
        "        value = self.encoders['enc_' + slot[:-8]](inputs[self.input_routes[slot]])",
        "        return value if slot in self.native_encoder_outputs else value.flatten(start_dim=1)",
        "",
    ]
    point = offset(forward.lineno, 0)
    edits.append((point, point, (newline.join(helper) + newline).encode()))
    for start, end, replacement in sorted(edits, reverse=True):
        raw = raw[:start] + replacement + raw[end:]
    try:
        candidate = raw.decode("utf-8")
        compile(candidate, "<starter-migration>", "exec")
    except (UnicodeDecodeError, SyntaxError) as error:
        _unsupported(f"候选源码无法编译：{error}")
    return candidate


def starter_migration_candidate(runtime, request, edit):
    """Build and execute a candidate before proposing a source transaction."""
    if request.get("variant") is not None:
        raise UserError("SOURCE_VARIANT_CONFLICT", "骨架迁移不能同时选择 Layer 变体。")
    model_ref = request.get("model") or ""
    module, separator, name = model_ref.partition(":")
    if not separator or name != "model" or not all(part.isidentifier() for part in module.split(".")):
        _unsupported("模型入口不是可识别的项目内 model 工厂。")
    root = Path(request["project_root"]).resolve()
    if request.get("code_snapshot_root") and Path(request["code_snapshot_root"]).resolve() != root:
        _unsupported("源码迁移不支持从独立代码快照读取模型。")
    relative = Path(*module.split(".")).with_suffix(".py")
    if any(part in {".venv", "venv", "site-packages", "target", ".git", "__pycache__", ".model-maven"} for part in relative.parts):
        _unsupported("模型文件不属于可编辑的项目源码。")
    lexical = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _unsupported("源码路径包含符号链接。")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
        source_bytes = resolved.read_bytes()
        if len(source_bytes) > 16 * 1024 * 1024:
            _unsupported("源码文件超过 16 MiB。")
        source = source_bytes.decode("utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        _unsupported(f"无法读取项目源码：{error}")
    candidate = migrate_starter_source(source)
    candidate_bytes = candidate.encode("utf-8")
    diff = "".join(difflib.unified_diff(source.splitlines(keepends=True), candidate.splitlines(keepends=True),
                                        fromfile=relative.as_posix(), tofile=relative.as_posix() + " (candidate)"))
    if len(candidate_bytes) + len(diff.encode("utf-8")) > 24 * 1024 * 1024:
        _unsupported("迁移预览超过 24 MiB。")
    parent_name, _, child_name = module.rpartition(".")
    absent = object()
    # Match a fresh worker's seed before importing the package and candidate.
    # The model loader must not reset it again between module code, component
    # construction, and the root factory.
    seed = request.get("seed") if request.get("seed") is not None else 42
    random.seed(seed)
    runtime.torch.manual_seed(seed)
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    if parent_name:
        try:
            parent = importlib.import_module(parent_name)
        except Exception as error:
            raise UserError("ENTRYPOINT_LOAD_FAILED", f"无法加载骨架父包：{error}") from error
        if module in sys.modules:
            _unsupported("父包在初始化时提前导入了目标模块，无法安全预览候选。")
    else:
        parent = None
    previous = sys.modules.get(module, absent)
    if previous is not absent:
        _unsupported("目标骨架模块已被提前加载，无法安全预览候选。")
    try:
        spec = importlib.util.find_spec(module)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
    except (ImportError, OSError, ValueError, AttributeError) as error:
        _unsupported(f"无法核对 Python 实际导入路径：{error}")
    if not spec or origin != resolved or type(spec.loader) is not importlib.machinery.SourceFileLoader:
        _unsupported("Python 实际导入路径或加载器与目标源码不一致，无法安全预览迁移。")
    candidate_module = importlib.util.module_from_spec(spec)
    parent_previous = getattr(parent, child_name, absent) if parent is not None else absent
    try:
        sys.modules[module] = candidate_module
        if parent is not None:
            setattr(parent, child_name, candidate_module)
        exec(compile(candidate, str(resolved), "exec"), candidate_module.__dict__)
        candidate_model, _, _, _ = runtime.load_model({**request, "variant": None}, reset_rng=False)
        validation = {"construction": "passed", "syntax": "passed", "forward": "not_run"}
        if request.get("batch"):
            observed = runtime.probe(candidate_model, {**request, "_disable_evidence_cache": True})
            validation.update({"forward": "passed", "outputs": observed["outputs"],
                               "mode": observed["observed_mode"], "forward_evidence": "executed"})
            if "batch_content" in observed:
                validation["batch_content"] = observed["batch_content"]
            if "batch_snapshot" in observed:
                validation["batch_snapshot"] = observed["batch_snapshot"]
        elif request.get("require_forward"):
            raise UserError("BATCH_REQUIRED", "迁移候选要求前向验证，但没有提供批次。", "使用 --batch module:factory。")
        groups = shared_state_groups(candidate_model)
        modules = runtime.inspect_model(candidate_model)["modules"]
        if request.get("replay_input_from") and request.get("batch"):
            runtime.entrypoint(request["batch"])
        manifest = runtime.source_manifest(str(root))
    finally:
        if previous is absent:
            sys.modules.pop(module, None)
        else:
            sys.modules[module] = previous
        if parent is not None:
            if parent_previous is absent and hasattr(parent, child_name):
                delattr(parent, child_name)
            elif parent_previous is not absent:
                setattr(parent, child_name, parent_previous)
    return {
        "source_patch": {"path": relative.as_posix(), "line": 1, "layer": "model",
                         "before_sha256": hashlib.sha256(source_bytes).hexdigest(),
                         "after_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                         "unified_diff": diff, "contents": candidate},
        "validation": validation,
        "source_manifest": manifest,
        "shared_state_evidence": {"scope": "in_memory_candidate_object_storage_v2", "groups": groups},
        "edit_effects": [{"action": "migrate_starter", "candidate_executed": True}],
        "modules": modules,
        "note": "候选已在当前 worker 内执行；确认提交后另行重建并按配置前向，失败时尝试回滚。现有检查点不会迁移；旧默认 head 的输入宽度不会随接线变化自动调整。",
    }
