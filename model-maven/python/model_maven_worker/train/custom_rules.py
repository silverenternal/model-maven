"""Pinned custom supervision callables from the project or selected environment."""

import hashlib
import importlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

from ..protocol import UserError
from ..run.file_digest import sha256_file


_MAX_PLUGIN_SOURCE = 8 * 1024 * 1024
_MAX_SUMMARY_DEPTH = 6
_MAX_SUMMARY_ITEMS = 256


def pin_plugin(project_root, kind, entrypoint, params, *, source_kind="project"):
    root = Path(project_root).resolve(strict=True)
    module_name, attribute = _split_entrypoint(entrypoint)
    if not isinstance(params, dict) or len(params) > 64:
        raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", "自定义规则参数必须是最多 64 项的 JSON 对象。")
    _validate_json(params)
    if len(json.dumps(params, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > 64 * 1024:
        raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", "自定义规则参数序列化后不得超过 64 KiB。")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if source_kind not in {"project", "environment"}:
        raise UserError("SUPERVISION_PLUGIN_INVALID", "规则源码来源必须是 project 或 environment。")
    target = _load_target(module_name, attribute)
    source = (_project_source(root, module_name) if source_kind == "project"
              else _environment_source(module_name, getattr(sys.modules.get(module_name), "__file__", None)))
    if not callable(target):
        raise UserError("SUPERVISION_PLUGIN_NOT_CALLABLE", f"自定义规则入口 {entrypoint} 不是 callable。")
    return {
        "protocol": "model-maven-supervision-v1",
        "entrypoint": entrypoint,
        "source": source,
        "params": params,
    }


def resolve_plugin(project_root, plugin, kind, mode, *, code_root=None):
    if not isinstance(plugin, dict) or plugin.get("protocol") != "model-maven-supervision-v1":
        raise UserError("SUPERVISION_PLUGIN_INVALID", "自定义监督规则协议版本无效。")
    module_name, attribute = _split_entrypoint(plugin.get("entrypoint"))
    recorded_source = plugin.get("source")
    if not isinstance(recorded_source, dict) or recorded_source.get("module") != module_name:
        raise UserError("SUPERVISION_PLUGIN_INVALID", "自定义监督规则入口与源码锚点不匹配。")
    if kind not in {"custom-validator", "custom-penalty"}:
        raise UserError("SUPERVISION_PLUGIN_INVALID", "不支持的自定义监督规则类型。")
    if kind == "custom-validator" and mode not in {"observe", "require"}:
        raise UserError("SUPERVISION_PLUGIN_INVALID", "custom-validator 只支持 observe 或 require。")
    if kind == "custom-penalty" and mode != "penalty":
        raise UserError("SUPERVISION_PLUGIN_INVALID", "custom-penalty 只支持 penalty。")
    root = Path(project_root).resolve(strict=True)
    source_root = Path(code_root).resolve(strict=True) if code_root is not None else root
    # Snapshot-backed training must not deliberately add the live project as
    # an import fallback. Existing selected import roots retain their order.
    if str(source_root) not in sys.path:
        sys.path.append(str(source_root))
    external = recorded_source.get("source_file") == "@environment:" + module_name
    expected = None if external else _recorded_source_path(root, source_root, recorded_source)
    try:
        # find_spec can import parent packages; verify the target before its
        # own import and check its identity again afterwards.
        spec = importlib.util.find_spec(module_name)
        origin = Path(spec.origin).resolve(strict=True) if spec and isinstance(spec.origin, str) else None
    except (ImportError, OSError, ValueError, AttributeError) as error:
        raise UserError("SUPERVISION_PLUGIN_SOURCE_UNAVAILABLE", f"无法核对自定义规则导入位置：{error}") from error
    if external:
        if _environment_source(module_name, origin) != recorded_source:
            raise UserError("SUPERVISION_PLUGIN_STALE", "所选环境中规则源码与已固定摘要不一致。",
                            "检查所选 Python 环境，或重新添加并复核规则。")
    elif origin != expected:
        raise UserError("SUPERVISION_PLUGIN_SOURCE_CONFLICT", "自定义规则的导入位置与已固定源码不一致。",
                        "检查所选模块搜索目录及环境；不会主动导入此目标模块。")
    target = _load_target(module_name, attribute)
    if not callable(target):
        raise UserError("SUPERVISION_PLUGIN_NOT_CALLABLE", f"自定义规则入口 {plugin.get('entrypoint')} 不是 callable。")
    current = (_environment_source(module_name, getattr(sys.modules.get(module_name), "__file__", None))
               if external else _project_source(root, module_name, code_root=code_root))
    if current != plugin.get("source"):
        raise UserError(
            "SUPERVISION_PLUGIN_STALE",
            f"自定义规则 {plugin.get('entrypoint')} 的源码已变化。",
            "重新添加规则并复核其协议行为后再预览或训练。",
            {"expected_source": plugin.get("source"), "current_source": current},
        )
    return target


def _recorded_source_path(project_root, source_root, source):
    relative = source.get("source_file") if isinstance(source, dict) else None
    expected_hash = source.get("sha256") if isinstance(source, dict) else None
    expected_size = source.get("size_bytes") if isinstance(source, dict) else None
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or Path(relative).is_absolute() or not relative.endswith(".py")
            or not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            or type(expected_size) is not int or not 0 < expected_size <= _MAX_PLUGIN_SOURCE):
        raise UserError("SUPERVISION_PLUGIN_INVALID", "自定义规则源码锚点无效。")
    try:
        source_root.relative_to(project_root)
        path = source_root / relative
        resolved = path.resolve(strict=True)
        info = resolved.stat()
        if resolved != path or not resolved.is_file() or info.st_size != expected_size:
            raise OSError("源码缺失、经过符号链接或大小变化")
        if sha256_file(resolved) != expected_hash:
            raise OSError("源码摘要变化")
    except (OSError, ValueError, RuntimeError) as error:
        raise UserError("SUPERVISION_PLUGIN_STALE", f"自定义规则源码无法核对：{error}",
                        "恢复已固定源码，或重新添加并确认规则。") from error
    return resolved


def evaluate_plugin(
    target, plugin, kind, value, batch, torch_module, *, training, sample_ids=None, axis=None
):
    plugin_value = value.detach().clone() if isinstance(value, torch_module.Tensor) else value
    if kind == "custom-penalty" and training and isinstance(value, torch_module.Tensor):
        plugin_value = value.clone()
    plugin_batch = _clone_batch(batch, torch_module)
    try:
        result = target(plugin_value, plugin_batch, plugin["params"])
    except Exception as error:
        raise UserError(
            "SUPERVISION_PLUGIN_FAILED",
            f"自定义规则 {plugin['entrypoint']} 执行失败：{type(error).__name__}: {error}",
            "本次规则未被视为通过；检查 callable 接口与当前实现块的输入。",
        ) from error
    if kind == "custom-validator":
        if not isinstance(result, dict) or set(result) - {"passed", "summary", "violations"} or type(result.get("passed")) is not bool:
            raise UserError(
                "SUPERVISION_PLUGIN_RESULT_INVALID",
                "validator 必须返回 {passed: bool, summary?: JSON object, violations?: bool Tensor}。",
            )
        summary = result.get("summary", {})
        if not isinstance(summary, dict):
            raise UserError("SUPERVISION_PLUGIN_RESULT_INVALID", "validator summary 必须是 JSON object。")
        _validate_summary(summary)
        mask = result.get("violations")
        violation_count = 0
        sample_localization = None
        if mask is not None:
            if (
                not isinstance(mask, torch_module.Tensor)
                or mask.dtype != torch_module.bool
                or not isinstance(value, torch_module.Tensor)
                or tuple(mask.shape) != tuple(value.shape)
                or mask.numel() > 1_000_000
            ):
                raise UserError("SUPERVISION_PLUGIN_RESULT_INVALID", "violations 必须是与标记端口同 shape 且不超过 1,000,000 项的 bool Tensor。")
            violation_count = int(mask.detach().sum().item())
            if axis is not None:
                if axis >= mask.ndim:
                    raise UserError("SUPERVISION_PLUGIN_RESULT_INVALID", "validator sample axis 超出 violations Tensor rank。")
                if sample_ids is None or len(sample_ids) != mask.shape[axis]:
                    raise UserError("SUPERVISION_SAMPLE_IDS_UNAVAILABLE", "validator 声明了样本轴，但当前批次缺少与轴长匹配的稳定 sample IDs。")
                per_sample = mask.detach().movedim(axis, 0).reshape(len(sample_ids), -1).sum(dim=1)
                violating = [
                    {"sample_id": sample_id, "violation_count": int(count)}
                    for sample_id, count in zip(sample_ids, per_sample.tolist()) if count
                ]
                sample_localization = {
                    "status": "localized", "stable_ids_available": True,
                    "sample_axis": axis, "sample_count": len(sample_ids),
                    "violating_sample_count": len(violating),
                    "violating_samples": violating[:50],
                    "violating_samples_truncated": len(violating) > 50,
                }
        if result["passed"] != (violation_count == 0):
            if mask is not None:
                raise UserError("SUPERVISION_PLUGIN_RESULT_INVALID", "validator passed 与 violations mask 互相矛盾。")
            violation_count = 0 if result["passed"] else 1
        return {
            "status": "measured",
            "passed": result["passed"],
            "shape": list(value.shape) if isinstance(value, torch_module.Tensor) else None,
            "element_count": int(mask.numel()) if mask is not None else 1,
            "violation_count": violation_count,
            "finite_count": 1,
            "nonfinite_count": 0,
            "sample_localization": sample_localization,
            "custom_summary": summary,
        }
    if not isinstance(result, torch_module.Tensor) or result.ndim != 0 or not result.is_floating_point():
        raise UserError("SUPERVISION_PLUGIN_RESULT_INVALID", "penalty callable 必须返回标量浮点 Tensor。")
    if not bool(torch_module.isfinite(result.detach()).all().item()):
        raise UserError("SUPERVISION_PLUGIN_RESULT_NONFINITE", "自定义 penalty 返回了 NaN 或无穷大。")
    if training and (not isinstance(value, torch_module.Tensor) or not result.requires_grad or not value.requires_grad):
        raise UserError("SUPERVISION_PENALTY_NOT_DIFFERENTIABLE", "自定义 penalty 或其绑定端口没有可用梯度。")
    return {
        "status": "measured",
        "passed": True,
        "shape": list(value.shape) if isinstance(value, torch_module.Tensor) else None,
        "element_count": 1,
        "violation_count": 0,
        "finite_count": 1,
        "nonfinite_count": 0,
        "custom_penalty_value": float(result.detach().item()),
        "penalty_tensor": result,
    }


def _clone_batch(value, torch_module, depth=0):
    if depth > 32:
        raise UserError("SUPERVISION_PLUGIN_BATCH_INVALID", "自定义规则 batch 嵌套超过 32 层。")
    if isinstance(value, torch_module.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_batch(child, torch_module, depth + 1) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_batch(child, torch_module, depth + 1) for child in value)
    if isinstance(value, list):
        return [_clone_batch(child, torch_module, depth + 1) for child in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise UserError("SUPERVISION_PLUGIN_BATCH_INVALID", f"自定义规则 batch 包含不支持类型：{type(value).__name__}。")


def _split_entrypoint(value):
    if not isinstance(value, str) or value.count(":") != 1 or len(value) > 512:
        raise UserError("SUPERVISION_PLUGIN_ENTRYPOINT_INVALID", "自定义规则入口须使用 module:callable。")
    module_name, attribute = value.split(":", 1)
    if (
        not module_name
        or not attribute
        or not all(part.isidentifier() for part in module_name.split("."))
        or not all(part.isidentifier() for part in attribute.split("."))
    ):
        raise UserError("SUPERVISION_PLUGIN_ENTRYPOINT_INVALID", "入口只允许 Python 模块及属性名称。")
    return module_name, attribute


def _load_target(module_name, attribute):
    try:
        target = importlib.import_module(module_name)
        for part in attribute.split("."):
            target = getattr(target, part)
    except Exception as error:
        raise UserError("SUPERVISION_PLUGIN_LOAD_FAILED", f"无法加载监督入口 {module_name}:{attribute}：{error}") from error
    return target


def _environment_source(module_name, filename):
    try:
        path = Path(filename).resolve(strict=True)
        info = path.stat()
        if path.suffix != ".py" or not path.is_file() or not 0 < info.st_size <= _MAX_PLUGIN_SOURCE:
            raise OSError("需要不超过 8 MiB 的可读 Python 源码")
        digest = sha256_file(path)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise UserError("SUPERVISION_PLUGIN_SOURCE_UNAVAILABLE",
                        f"无法读取所选环境中的规则源码：{error}") from error
    return {"module": module_name, "sha256": digest, "size_bytes": info.st_size,
            "source_file": "@environment:" + module_name}


def _project_source(root, module_name, *, code_root=None):
    module = sys.modules.get(module_name)
    filename = getattr(module, "__file__", None)
    if not isinstance(filename, str) or not filename.endswith(".py"):
        raise UserError("SUPERVISION_PLUGIN_SOURCE_UNAVAILABLE", "自定义规则必须位于项目内普通 Python 源文件。")
    candidate = Path(filename)
    try:
        if candidate.is_symlink():
            raise OSError("源码是符号链接")
        resolved = candidate.resolve(strict=True)
        if code_root is not None:
            snapshot_root = Path(code_root).resolve(strict=True)
            snapshot_root.relative_to(root)
            relative = resolved.relative_to(snapshot_root)
        else:
            relative = resolved.relative_to(root)
        info = resolved.stat()
        if not resolved.is_file() or info.st_size <= 0 or info.st_size > _MAX_PLUGIN_SOURCE:
            raise OSError("源码文件类型或大小不受支持")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        raise UserError(
            "SUPERVISION_PLUGIN_SOURCE_UNAVAILABLE",
            "自定义规则源码必须是项目内可读的普通 .py 文件，大小不超过 8 MiB。",
        ) from error
    return {"module": module_name, "sha256": digest, "size_bytes": info.st_size, "source_file": relative.as_posix()}


def _validate_json(value, depth=0):
    if depth > _MAX_SUMMARY_DEPTH:
        raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", "自定义规则 JSON 嵌套最多 6 层。")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", "自定义规则参数不得包含非有限数字。")
        return
    if isinstance(value, list):
        if len(value) > _MAX_SUMMARY_ITEMS:
            raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", "自定义规则数组最多包含 256 项。")
        for item in value:
            _validate_json(item, depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_SUMMARY_ITEMS or any(not isinstance(key, str) for key in value):
            raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", "自定义规则对象最多包含 256 个字符串键。")
        for item in value.values():
            _validate_json(item, depth + 1)
        return
    raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", f"不支持自定义规则参数类型：{type(value).__name__}。")


def _validate_summary(value, depth=0):
    try:
        _validate_json(value, depth)
        if len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > 16 * 1024:
            raise UserError("SUPERVISION_PLUGIN_RESULT_INVALID", "validator summary 超过 16 KiB。")
    except UserError as error:
        raise UserError("SUPERVISION_PLUGIN_RESULT_INVALID", f"validator summary 无效：{str(error)}") from error
