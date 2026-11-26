"""Declared model port contracts checked against bounded runtime observations."""

import json
import re
import hashlib
import hmac
from pathlib import Path

from ..protocol import UserError


_ROOTS = {"inputs", "outputs"}
_TENSOR_FIELDS = {"rank", "dtype", "layout", "memory_format", "shape"}
_PATH_TOKEN = re.compile(r'(?:\.([^\.\[\]]+)|\[([0-9]+)\]|\[("(?:[^"\\]|\\.)*")\])')
_UNOBSERVED = object()


def load_contract(request):
    """Read and validate one bounded contract before model or data factories."""
    request.pop("_contract_content_sha256", None)
    contract_path = request.get("contract")
    expected = request.get("if_contract_sha256")
    if expected is not None and (not contract_path or not isinstance(expected, str)
                                 or not re.fullmatch(r"[0-9a-f]{64}", expected)):
        raise UserError("MODEL_CONTRACT_IDENTITY_INVALID", "契约身份条件必须与契约文件一起提供，并且是 64 位小写 SHA-256。")
    if not contract_path:
        return None
    path = Path(contract_path)
    if not path.is_absolute():
        path = Path(request.get("project_root", ".")) / path
    try:
        with path.open("rb") as stream:
            raw = stream.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise UserError("MODEL_CONTRACT_INVALID", "模型契约超过 8 MiB 上限。")
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and not hmac.compare_digest(digest, expected):
            raise UserError("MODEL_CONTRACT_CHANGED", "契约内容与确认时不同，未执行受该契约约束的前向。",
                            "重新查看契约并确认；此前已发生的用户代码副作用不会回滚。")
        contract = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("MODEL_CONTRACT_INVALID", f"无法读取模型契约 {str(path)!r}：{error}",
                        "使用有效的 UTF-8 JSON 契约文件；契约只参与检查，不会修改模型或项目配置。") from error
    contract = validate_contract_definition(contract)
    request["_contract_content_sha256"] = digest
    return contract


def validate_contract_definition(contract):
    if not isinstance(contract, dict):
        raise UserError("MODEL_CONTRACT_INVALID", "模型契约必须是 JSON 对象。")
    unknown = set(contract) - {"version", "inputs", "outputs", "modules", "symbols", "static"}
    if unknown:
        raise UserError(
            "MODEL_CONTRACT_INVALID",
            f"模型契约包含未知顶层字段：{', '.join(sorted(unknown))}。",
        )
    if contract.get("version", 1) != 1:
        raise UserError("MODEL_CONTRACT_INVALID", "当前只支持模型契约版本 1。")
    if "static" in contract and type(contract["static"]) is not bool:
        raise UserError("MODEL_CONTRACT_INVALID", "static 必须是布尔值。")
    symbols = contract.get("symbols", {})
    if not isinstance(symbols, dict):
        raise UserError("MODEL_CONTRACT_INVALID", "symbols 必须是对象。")
    for name, bounds in symbols.items():
        if not isinstance(name, str) or not name or not isinstance(bounds, dict):
            raise UserError("MODEL_CONTRACT_INVALID", "每个符号维度都必须有名称和范围对象。")
        if set(bounds) - {"min", "max", "equal"}:
            raise UserError("MODEL_CONTRACT_INVALID", f"符号维度 {name} 含未知范围字段。")
        for key in ("min", "max", "equal"):
            value = bounds.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise UserError("MODEL_CONTRACT_INVALID", f"符号维度 {name}.{key} 必须是非负整数。")
        if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
            raise UserError("MODEL_CONTRACT_INVALID", f"符号维度 {name} 的最小值大于最大值。")
        if "equal" in bounds and (
            ("min" in bounds and bounds["equal"] < bounds["min"])
            or ("max" in bounds and bounds["equal"] > bounds["max"])
        ):
            raise UserError("MODEL_CONTRACT_INVALID", f"符号维度 {name}.equal 不在声明范围内。")
    for root in ("inputs", "outputs"):
        ports = contract.get(root, {})
        if not isinstance(ports, dict):
            raise UserError("MODEL_CONTRACT_INVALID", f"{root} 必须是端口路径到约束的对象。")
        for path, spec in ports.items():
            if not isinstance(path, str) or not path or not isinstance(spec, dict):
                raise UserError("MODEL_CONTRACT_INVALID", f"{root} 端口声明格式无效。")
            if set(spec) - _TENSOR_FIELDS:
                raise UserError("MODEL_CONTRACT_INVALID", f"端口 {root}.{path} 含未知约束字段。")
            if "rank" in spec and (type(spec["rank"]) is not int or spec["rank"] < 0):
                raise UserError("MODEL_CONTRACT_INVALID", f"端口 {root}.{path}.rank 必须是非负整数。")
            if "dtype" in spec and not isinstance(spec["dtype"], str):
                raise UserError("MODEL_CONTRACT_INVALID", f"端口 {root}.{path}.dtype 必须是字符串。")
            if "layout" in spec and not isinstance(spec["layout"], str):
                raise UserError("MODEL_CONTRACT_INVALID", f"端口 {root}.{path}.layout 必须是字符串。")
            if "memory_format" in spec and spec["memory_format"] not in {
                "contiguous", "channels_last", "channels_last_3d"
            }:
                raise UserError("MODEL_CONTRACT_INVALID", f"端口 {root}.{path}.memory_format 不受支持。")
            if "shape" in spec:
                shape = spec["shape"]
                if not isinstance(shape, list):
                    raise UserError("MODEL_CONTRACT_INVALID", f"端口 {root}.{path}.shape 必须是数组。")
                for dim in shape:
                    if dim is None or isinstance(dim, str):
                        if isinstance(dim, str) and dim not in symbols:
                            raise UserError("MODEL_CONTRACT_INVALID", f"端口 {root}.{path} 使用了未声明符号 {dim}。")
                    elif type(dim) is not int or dim < 0:
                        raise UserError("MODEL_CONTRACT_INVALID", f"端口 {root}.{path}.shape 维度无效。")
                if "rank" in spec and spec["rank"] != len(shape):
                    raise UserError("MODEL_CONTRACT_INVALID", f"端口 {root}.{path} 的 rank 与 shape 长度冲突。")
    modules = contract.get("modules", {})
    if not isinstance(modules, dict):
        raise UserError("MODEL_CONTRACT_INVALID", "modules 必须是模块路径到端口声明的对象。")
    for module_path, ports in modules.items():
        if not isinstance(module_path, str) or not (module_path == "model" or module_path.startswith("model.")) or not isinstance(ports, dict):
            raise UserError("MODEL_CONTRACT_INVALID", "modules 的键必须是 model 或 model.* 路径且值为端口声明对象。")
        if set(ports) - {"inputs", "outputs"}:
            raise UserError("MODEL_CONTRACT_INVALID", f"模块契约 {module_path} 只支持 inputs 和 outputs。")
        validate_contract_definition({"version": 1, "symbols": symbols,
                                      "inputs": ports.get("inputs", {}),
                                      "outputs": ports.get("outputs", {})})
    return contract


def _resolve(tree, dotted_path):
    """Resolve tensor-tree paths such as inputs.args[0] without arbitrary eval."""
    if not isinstance(dotted_path, str) or not dotted_path:
        return _UNOBSERVED
    root = next((name for name in _ROOTS if dotted_path == name or dotted_path.startswith((name + ".", name + "["))), None)
    if root is None:
        return _UNOBSERVED
    current = tree.get(root, _UNOBSERVED) if isinstance(tree, dict) else _UNOBSERVED
    tail = dotted_path[len(root):]
    index = 0
    while index < len(tail):
        match = _PATH_TOKEN.match(tail, index)
        if match is None:
            return _UNOBSERVED
        name, item_index, quoted = match.groups()
        if quoted is not None:
            try:
                name = json.loads(quoted)
            except ValueError:
                return _UNOBSERVED
        if item_index is not None:
            if not isinstance(current, dict) or current.get("kind") not in {"list", "tuple"}:
                return _UNOBSERVED
            items = current.get("items")
            item_index = int(item_index)
            current = items[item_index] if isinstance(items, list) and 0 <= item_index < len(items) else _UNOBSERVED
        else:
            if (not isinstance(current, dict) or current.get("kind") != "dict"
                    or current.get("observed_keys_all_strings") is not True):
                return _UNOBSERVED
            current = current.get("items", {}).get(name, _UNOBSERVED)
        index = match.end()
    return current


def check_contract(contract, input_tree, output_tree, calls=None, _bindings=None):
    contract = validate_contract_definition(contract)
    trees = {"inputs": input_tree, "outputs": output_tree}
    bindings = _bindings if _bindings is not None else {}
    checks = []

    def check(path, spec):
        actual = _resolve(trees, path)
        if actual is _UNOBSERVED:
            checks.append({"path": path, "status": "unobserved", "expected": spec, "actual": None,
                           "coverage": "本次有界结构摘要无法定位该端口；它可能不存在、被截断或因键类型歧义无法表达。"})
            return
        if not isinstance(actual, dict) or actual.get("kind") != "tensor":
            checks.append({"path": path, "status": "unsupported", "expected": spec, "actual": actual,
                           "coverage": "端口不是可检查的 Tensor；仅报告嵌套结构事实。"})
            return
        shape = actual.get("shape")
        mismatches = []
        unknowns = []
        for field in ("rank", "dtype", "layout", "memory_format"):
            if field not in spec:
                continue
            observed = actual.get(field)
            expected = spec[field]
            if field == "dtype":
                expected = expected.removeprefix("torch.")
                observed = observed.removeprefix("torch.") if isinstance(observed, str) else observed
            if field == "rank":
                observed = len(shape) if isinstance(shape, list) else None
            if observed is None:
                unknowns.append(field)
            elif observed != expected:
                mismatches.append({"field": field, "expected": expected, "actual": observed})
        expected_shape = spec.get("shape")
        if expected_shape is not None:
            if not isinstance(shape, list):
                unknowns.append("shape")
            elif len(shape) != len(expected_shape):
                mismatches.append({"field": "shape", "expected": expected_shape, "actual": shape})
            else:
                for index, (expected, observed) in enumerate(zip(expected_shape, shape)):
                    if expected is None:
                        continue
                    if isinstance(expected, int):
                        if observed != expected:
                            mismatches.append({"field": f"shape[{index}]", "expected": expected, "actual": observed})
                        continue
                    bounds = contract.get("symbols", {}).get(expected, {})
                    if expected in bindings and bindings[expected] != observed:
                        mismatches.append({"field": f"shape[{index}]", "expected": bindings[expected],
                                           "actual": observed, "symbol": expected})
                    else:
                        bindings[expected] = observed
                    for bound, relation in (("min", lambda v, b: v >= b),
                                            ("max", lambda v, b: v <= b),
                                            ("equal", lambda v, b: v == b)):
                        if bound in bounds and not relation(observed, bounds[bound]):
                            mismatches.append({"field": f"shape[{index}]", "expected": bounds,
                                               "actual": observed, "symbol": expected})
        status = "failed" if mismatches else ("unsupported" if unknowns else "observed-runtime")
        checks.append({"path": path, "status": status, "expected": spec,
                       "actual": actual, "mismatches": mismatches, "unknown_fields": unknowns,
                       "coverage": "仅本次批次和已执行分支的真实前向观测。"})

    for root in ("inputs", "outputs"):
        for path, spec in contract.get(root, {}).items():
            full_path = path if path == root or path.startswith((root + ".", root + "[")) else (root + path if path.startswith("[") else f"{root}.{path}")
            check(full_path, spec)
    for module_path, ports in contract.get("modules", {}).items():
        matches = [call for call in (calls or []) if module_path in call.get("alias_paths", call.get("paths", []))]
        if not matches:
            checks.append({"path": module_path, "status": "unobserved", "expected": ports, "actual": None,
                           "coverage": "本次前向没有执行该模块路径。"})
            continue
        for call in matches:
            local_contract = {
                "version": 1,
                "symbols": contract.get("symbols", {}),
                "inputs": ports.get("inputs", {}),
                "outputs": ports.get("outputs", {}),
            }
            local_report = check_contract(
                local_contract,
                call.get("inputs", {}),
                call.get("outputs", {}),
                _bindings=bindings,
            )
            invocation = call.get("invocation_id", call.get("invocation_index", "unknown"))
            for local_check in local_report["checks"]:
                local_check["path"] = f"{module_path}#{invocation}.{local_check['path']}"
                local_check["invocation_id"] = invocation
                checks.append(local_check)
    failed = any(item["status"] == "failed" for item in checks)
    partial = not checks or any(item["status"] in {"unsupported", "unobserved"} for item in checks)
    return {
        "status": "failed" if failed else ("partial" if partial else "observed-runtime"),
        "checks": checks,
        "symbol_bindings": bindings,
        "coverage": ("没有声明可检查的端口约束；本次没有验证契约。" if not checks else
                     "已检查声明端口在本次真实前向中的可观察范围；不证明未执行的动态分支。"),
    }
