"""Persistent, explicitly authored semantic marks for observed model ports."""

import hashlib
import functools
import stat
from contextlib import contextmanager
import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath

from ..protocol import UserError


_FORMAT_VERSION = 1
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_MARKS = 512
_MAX_RULES_PER_MARK = 32
_MARK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_BATCH_PATH_TOKEN = re.compile(r"(?:\.([A-Za-z_][A-Za-z0-9_-]*)|\[([0-9]+)\])")


def _state_revision(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _state_path(project_root):
    root = Path(project_root).resolve(strict=True)
    state_dir = root / ".model-maven"
    try:
        info = state_dir.lstat()
    except FileNotFoundError:
        state_dir.mkdir(mode=0o700, exist_ok=True)
        if state_dir.is_symlink() or not state_dir.is_dir():
            raise UserError("SEMANTIC_STATE_PATH_INVALID", "语义标记目录必须是项目内的普通目录。")
    else:
        if state_dir.is_symlink() or not state_dir.is_dir():
            raise UserError(
                "SEMANTIC_STATE_PATH_INVALID",
                "语义标记目录必须是项目内的普通目录，不能是符号链接。",
            )
        if info.st_mode & 0o077:
            try:
                state_dir.chmod(0o700)
            except OSError as error:
                raise UserError("SEMANTIC_STATE_UNAVAILABLE", str(error)) from error
    return root, state_dir / "semantic-marks.json"


@contextmanager
def _state_lock(path):
    """Serialize cooperating semantic-state writers across processes."""
    lock_path = path.parent / "semantic-marks.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise UserError("SEMANTIC_STATE_LOCK_UNAVAILABLE", f"无法打开语义状态锁：{error}") from error
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or lock_path.is_symlink():
            raise UserError("SEMANTIC_STATE_PATH_INVALID", "语义状态锁必须是普通文件。")
        if os.name == "nt":
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError as error:
        raise UserError("SEMANTIC_STATE_LOCK_UNAVAILABLE", f"无法使用语义状态锁：{error}") from error
    finally:
        os.close(fd)


def _serialized_mutation(function):
    @functools.wraps(function)
    def guarded(project_root, *args, **kwargs):
        _, path = _state_path(project_root)
        with _state_lock(path):
            return function(project_root, *args, **kwargs)
    return guarded


def _source_anchor(project_root, implementation):
    if not isinstance(implementation, dict):
        raise UserError(
            "SEMANTIC_SOURCE_UNAVAILABLE",
            "该模块没有可用的项目源码锚点；无法持久标记。",
            "将实现放入项目源码后重新 probe；外部包路径不作为可移植锚点。",
        )
    relative = implementation.get("source_file")
    qualified_type = implementation.get("qualified_type")
    if not isinstance(relative, str) or not relative or not isinstance(qualified_type, str):
        raise UserError("SEMANTIC_SOURCE_UNAVAILABLE", "模块实现来源信息不完整。")
    root = Path(project_root).resolve(strict=True)
    candidate = root / relative
    try:
        info = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise UserError(
            "SEMANTIC_SOURCE_UNAVAILABLE",
            "模块源码不存在、不可读或已离开项目根目录。",
            "源码锚点必须位于项目内；检查模型构造和源码路径后重新 probe。",
        ) from error
    if candidate.is_symlink() or not resolved.is_file() or info.st_size > _MAX_SOURCE_BYTES:
        raise UserError(
            "SEMANTIC_SOURCE_UNAVAILABLE",
            "模块源码是符号链接、非普通文件或超过 8 MiB，不能建立持久锚点。",
        )
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    expected = implementation.get("source_sha256")
    if expected != digest:
        raise UserError(
            "SEMANTIC_SOURCE_CHANGED",
            "Probe 之后模块源码已变化，拒绝把标记绑定到旧观测。",
            "重新 probe 后再添加语义标记。",
        )
    return {
        "source_file": resolved.relative_to(root).as_posix(),
        "source_sha256": digest,
        "qualified_type": qualified_type[:512],
    }


def _read_state(path):
    if not path.exists():
        return {"version": _FORMAT_VERSION, "marks": []}
    if path.is_symlink() or not path.is_file():
        raise UserError("SEMANTIC_STATE_PATH_INVALID", "语义标记存储必须是普通文件。")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("SEMANTIC_STATE_INVALID", f"无法读取语义标记存储：{error}") from error
    if (
        not isinstance(value, dict)
        or value.get("version") != _FORMAT_VERSION
        or not isinstance(value.get("marks"), list)
        or len(value.get("marks", [])) > _MAX_MARKS
    ):
        raise UserError(
            "SEMANTIC_STATE_INVALID",
            "语义标记存储格式版本不受支持或内容损坏；未覆盖原文件。",
        )
    seen = set()
    for mark in value["marks"]:
        anchor = mark.get("anchor") if isinstance(mark, dict) else None
        context = mark.get("binding_context") if isinstance(mark, dict) else None
        source_file = anchor.get("source_file") if isinstance(anchor, dict) else None
        source_path = PurePosixPath(source_file) if isinstance(source_file, str) else None
        if (
            not isinstance(mark, dict)
            or not isinstance(mark.get("id"), str)
            or not _MARK_ID.fullmatch(mark["id"])
            or mark["id"] in seen
            or not isinstance(mark.get("role"), str)
            or not isinstance(mark.get("meaning"), str)
            or not isinstance(mark.get("module_path"), str)
            or not isinstance(mark.get("port"), str)
            or not isinstance(anchor, dict)
            or source_path is None
            or source_path.is_absolute()
            or ".." in source_path.parts
            or not isinstance(anchor.get("source_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", anchor["source_sha256"])
            or not isinstance(anchor.get("qualified_type"), str)
            or not isinstance(context, dict)
            or not isinstance(context.get("model"), str)
            or not isinstance(mark.get("observation"), dict)
            or not isinstance(mark.get("rules", []), list)
            or len(mark.get("rules", [])) > _MAX_RULES_PER_MARK
        ):
            raise UserError(
                "SEMANTIC_STATE_INVALID",
                "语义标记存储包含格式错误、重复 ID 或不安全源码路径；未修改原文件。",
            )
        seen.add(mark["id"])
        rule_ids = set()
        for rule in mark.get("rules", []):
            if (
                not isinstance(rule, dict)
                or not isinstance(rule.get("id"), str)
                or not _MARK_ID.fullmatch(rule["id"])
                or rule["id"] in rule_ids
                or rule.get("kind") not in {"finite", "range", "batch-alignment", "custom-validator", "custom-penalty"}
                or rule.get("mode") not in {"observe", "require", "penalty"}
                or any(
                    rule.get(key) is not None
                    and (
                        isinstance(rule.get(key), bool)
                        or not isinstance(rule.get(key), (int, float))
                        or not math.isfinite(rule[key])
                    )
                    for key in ("min", "max", "weight")
                )
                or (rule.get("kind") == "finite" and (rule.get("min") is not None or rule.get("max") is not None))
                or (rule.get("kind") == "range" and rule.get("min") is None and rule.get("max") is None)
                or (
                    rule.get("kind") == "batch-alignment"
                    and (
                        not isinstance(rule.get("target_path"), str)
                        or not _valid_batch_target_path(rule["target_path"])
                        or isinstance(rule.get("axis"), bool)
                        or not isinstance(rule.get("axis"), int)
                        or not 0 <= rule["axis"] <= 16
                    )
                )
                or (
                    rule.get("kind") != "batch-alignment"
                    and rule.get("target_path") is not None
                )
                or (
                    rule.get("kind") in {"finite", "range", "custom-validator"}
                    and rule.get("axis") is not None
                    and (
                        isinstance(rule.get("axis"), bool)
                        or not isinstance(rule.get("axis"), int)
                        or not 0 <= rule["axis"] <= 16
                    )
                )
                or (rule.get("min") is not None and rule.get("max") is not None and rule["min"] > rule["max"])
                or (rule.get("mode") == "penalty" and (rule.get("weight") is None or rule["weight"] <= 0))
                or (rule.get("mode") != "penalty" and rule.get("weight") is not None)
                or (rule.get("kind") in {"custom-validator", "custom-penalty"} and not _valid_custom_plugin(rule.get("plugin"), rule.get("kind"), rule.get("mode")))
                or (rule.get("kind") not in {"custom-validator", "custom-penalty"} and rule.get("plugin") is not None)
            ):
                raise UserError("SEMANTIC_STATE_INVALID", "监督规则存储格式无效或 ID 重复。")
            rule_ids.add(rule["id"])
    return value


def _write_state(path, state):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".semantic-marks-", dir=path.parent)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def binding_context(probe_data, variant, source_manifest):
    origin = probe_data.get("origin", {}) if isinstance(probe_data, dict) else {}

    def digest(value):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    model_ref = origin.get("model")
    if not isinstance(model_ref, str) or not model_ref:
        raise UserError("SEMANTIC_MODEL_IDENTITY_MISSING", "无法确认标记所绑定的模型入口。")
    return {
        "model": model_ref,
        "model_params_sha256": digest(origin.get("params", {})),
        "model_injections_sha256": digest(origin.get("model_injections", {})),
        "variant_sha256": digest(variant) if isinstance(variant, dict) else None,
        "source_manifest_sha256": digest(source_manifest),
    }


@_serialized_mutation
def add_from_probe(
    project_root,
    probe_data,
    *,
    module_path,
    port,
    role,
    meaning,
    mark_id=None,
    replace_existing=False,
    context=None,
):
    """Bind a human-authored mark only to a module port present in this probe."""
    module_path = module_path.strip() if isinstance(module_path, str) else ""
    port = port.strip() if isinstance(port, str) else ""
    role = role.strip() if isinstance(role, str) else ""
    meaning = meaning.strip() if isinstance(meaning, str) else ""
    if not module_path.startswith("model.") or len(module_path) > 512:
        raise UserError("SEMANTIC_MODULE_PATH_INVALID", "实现块须使用完整模型树路径，例如 model.encoder.projection。")
    if not (port.startswith("inputs") or port.startswith("outputs")) or len(port) > 512:
        raise UserError("SEMANTIC_PORT_INVALID", "端口须是 Probe 显示的 inputs/outputs 路径。")
    if not role or len(role) > 80 or any(ord(char) < 32 for char in role):
        raise UserError("SEMANTIC_ROLE_INVALID", "语义角色须为 1 到 80 个可见字符。")
    if not meaning or len(meaning) > 512 or any(ord(char) < 32 and char not in "\t\n" for char in meaning):
        raise UserError("SEMANTIC_MEANING_INVALID", "语义说明须为 1 到 512 个字符。")
    if not isinstance(probe_data, dict) or not isinstance(probe_data.get("probe_id"), str):
        raise UserError("SEMANTIC_OBSERVATION_REQUIRED", "需要一次真实模型 Probe 观测才能标记。")
    if probe_data.get("trace_truncated") is True:
        raise UserError("SEMANTIC_OBSERVATION_TRUNCATED", "本次调用轨迹已截断；不能确认目标实现块覆盖。", "缩小模型路径或提高调用观测预算后重新 Probe。")

    matches = []
    for call in probe_data.get("calls", []):
        paths = call.get("paths", []) if isinstance(call, dict) else []
        ports = call.get("input_output_paths", []) if isinstance(call, dict) else []
        if module_path in paths and port in ports:
            matches.append(call)
    if not matches:
        raise UserError(
            "SEMANTIC_PORT_NOT_OBSERVED",
            f"本次 Probe 没有观察到 {module_path} 的端口 {port}。",
            "使用 Probe 展示的真实模块路径和端口；未观察路径不会被标记为通过。",
        )
    anchors = {
        json.dumps(call.get("implementation"), sort_keys=True)
        for call in matches
    }
    if len(anchors) != 1:
        raise UserError(
            "SEMANTIC_ANCHOR_AMBIGUOUS",
            "目标路径对应多个不同实现来源，无法安全建立单一标记。",
            "选择更具体的模块路径后重新 Probe。",
        )
    implementation = matches[0].get("implementation")
    source = _source_anchor(project_root, implementation)
    if not isinstance(context, dict):
        raise UserError("SEMANTIC_MODEL_IDENTITY_MISSING", "无法建立模型配置与变体绑定指纹。")
    mark_id = mark_id or uuid.uuid4().hex[:16]
    if not isinstance(mark_id, str) or not _MARK_ID.fullmatch(mark_id):
        raise UserError("SEMANTIC_ID_INVALID", "标记 ID 仅支持 1 到 80 位字母、数字、点、下划线或连字符。")

    root, path = _state_path(project_root)
    state = _read_state(path)
    if len(state["marks"]) >= _MAX_MARKS:
        raise UserError("SEMANTIC_MARK_LIMIT", f"每个项目最多保存 {_MAX_MARKS} 个语义标记。")
    existing_index = next(
        (index for index, item in enumerate(state["marks"]) if item.get("id") == mark_id),
        None,
    )
    if existing_index is None and replace_existing:
        raise UserError("SEMANTIC_MARK_NOT_FOUND", f"找不到要重新绑定的语义标记：{mark_id}。")
    if existing_index is not None and not replace_existing:
        raise UserError("SEMANTIC_ID_EXISTS", f"语义标记 ID 已存在：{mark_id}。")
    mark = {
        "id": mark_id,
        "role": role,
        "meaning": meaning,
        "model": context["model"],
        "binding_context": context,
        "module_path": module_path,
        "port": port,
        "anchor": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "observation": {
            "probe_id": probe_data["probe_id"],
            "batch_id": probe_data.get("batch_id"),
            "batch_identity_stable": probe_data.get("batch_identity_stable") is True,
            "invocation_ids": sorted({call.get("invocation_id") for call in matches if call.get("invocation_id")}),
        },
        "rules": state["marks"][existing_index].get("rules", []) if existing_index is not None else [],
    }
    if existing_index is not None:
        mark["created_at"] = state["marks"][existing_index].get("created_at", mark["created_at"])
        mark["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["marks"][existing_index] = mark
    else:
        state["marks"].append(mark)
    _write_state(path, state)
    return {
        "mark": mark,
        "status": "rebound" if existing_index is not None else "current",
        "path": path.relative_to(root).as_posix(),
    }


def list_marks(project_root):
    root, path = _state_path(project_root)
    state = _read_state(path)
    reports = []
    for mark in state["marks"]:
        anchor = mark.get("anchor", {})
        source_path = root / str(anchor.get("source_file", ""))
        status = "current"
        reason = None
        try:
            if source_path.is_symlink() or not source_path.is_file():
                raise OSError("源码文件不存在或不再是普通文件")
            resolved = source_path.resolve(strict=True)
            resolved.relative_to(root)
            if resolved.stat().st_size > _MAX_SOURCE_BYTES:
                raise OSError("源码文件超过指纹大小上限")
            actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if actual != anchor.get("source_sha256"):
                status = "stale"
                reason = "源文件指纹已变化，需重新 probe 并手动绑定。"
        except OSError:
            status = "stale"
            reason = "锚定源码不可访问，需重新 probe 并手动绑定。"
        reports.append({**mark, "binding_status": status, "stale_reason": reason})
    return {"marks": reports, "count": len(reports), "path": path.relative_to(root).as_posix(), "state_revision": _state_revision(state)}


def audit_marks(project_root, probe_data, *, context):
    """Compare saved marks with a fresh full-model observation and config identity."""
    listed = list_marks(project_root)
    calls = probe_data.get("calls", []) if isinstance(probe_data, dict) else []
    audited = []
    for mark in listed["marks"]:
        report = dict(mark)
        if report["binding_status"] == "stale":
            audited.append(report)
            continue
        matches = [
            call for call in calls
            if report["module_path"] in call.get("paths", [])
            and report["port"] in call.get("input_output_paths", [])
        ]
        if not matches:
            if probe_data.get("trace_truncated") is True:
                report["binding_status"] = "unverified"
                report["stale_reason"] = "本次轨迹截断，未能确认该标记仍绑定当前模型；需扩大探测覆盖。"
            else:
                report["binding_status"] = "stale"
                report["stale_reason"] = "当前模型观测不到原模块路径/端口；需手动重新绑定。"
            audited.append(report)
            continue
        if any(call.get("implementation") != report.get("anchor") for call in matches):
            report["binding_status"] = "stale"
            report["stale_reason"] = "模块路径现在对应不同实现；需手动重新绑定。"
        elif report.get("binding_context") != context:
            report["binding_status"] = "stale"
            report["stale_reason"] = "模型入口、参数、注入或变体已变化；需重新确认语义绑定。"
        audited.append(report)
    return {
        "marks": audited,
        "count": len(audited),
        "probe_id": probe_data.get("probe_id"),
        "audit_status": "incomplete" if probe_data.get("trace_truncated") is True else "observed-runtime",
        "coverage": "只对本次已执行分支、当前批次和模块调用做绑定重验；未覆盖分支仍未知。",
    }


@_serialized_mutation
def remove_mark(project_root, mark_id):
    root, path = _state_path(project_root)
    state = _read_state(path)
    remaining = [mark for mark in state["marks"] if mark.get("id") != mark_id]
    if len(remaining) == len(state["marks"]):
        raise UserError("SEMANTIC_MARK_NOT_FOUND", f"找不到语义标记：{mark_id}。")
    state["marks"] = remaining
    _write_state(path, state)
    return {"removed": mark_id, "count": len(remaining), "path": path.relative_to(root).as_posix()}


def _valid_batch_target_path(path):
    if path == "batch" or not path.startswith("batch.") and not path.startswith("batch[") or len(path) > 512:
        return False
    tail = path[5:]
    position = 0
    while position < len(tail):
        match = _BATCH_PATH_TOKEN.match(tail, position)
        if match is None:
            return False
        position = match.end()
    return position == len(tail)


def _valid_custom_plugin(plugin, kind, mode):
    if not isinstance(plugin, dict) or set(plugin) != {"protocol", "entrypoint", "source", "params"}:
        return False
    source = plugin.get("source")
    entrypoint = plugin.get("entrypoint")
    if (
        plugin.get("protocol") != "model-maven-supervision-v1"
        or not isinstance(entrypoint, str)
        or entrypoint.count(":") != 1
        or not isinstance(source, dict)
        or set(source) != {"module", "sha256", "size_bytes", "source_file"}
        or not isinstance(source.get("module"), str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", source["module"])
        or not isinstance(source.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        or not isinstance(source.get("source_file"), str)
        or not source.get("source_file")
        or PurePosixPath(source["source_file"]).is_absolute()
        or ".." in PurePosixPath(source["source_file"]).parts
        or type(source.get("size_bytes")) is not int
        or not 0 < source["size_bytes"] <= _MAX_SOURCE_BYTES
        or not isinstance(plugin.get("params"), dict)
        or len(plugin["params"]) > 64
    ):
        return False
    module, target = entrypoint.split(":", 1)
    if module != source["module"] or not all(part.isidentifier() for part in target.split(".")):
        return False
    return mode in {"observe", "require"} if kind == "custom-validator" else kind == "custom-penalty" and mode == "penalty"


@_serialized_mutation
def add_rule(
    project_root,
    mark_id,
    *,
    kind,
    mode,
    minimum=None,
    maximum=None,
    weight=None,
    target_path=None,
    axis=None,
    rule_id=None,
    plugin=None,
):
    if kind not in {"finite", "range", "batch-alignment", "custom-validator", "custom-penalty"}:
        raise UserError("SUPERVISION_RULE_KIND_INVALID", "规则 kind 目前支持 finite、range 或 batch-alignment。")
    if mode not in {"observe", "require", "penalty"}:
        raise UserError("SUPERVISION_RULE_MODE_INVALID", "规则 mode 目前只支持 observe、require 或 penalty。")
    for name, value in (("min", minimum), ("max", maximum), ("weight", weight)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise UserError("SUPERVISION_RULE_VALUE_INVALID", f"规则 {name} 必须是有限数字。")
    if kind == "finite" and (minimum is not None or maximum is not None):
        raise UserError("SUPERVISION_RULE_VALUE_INVALID", "finite 规则不接受 min/max。")
    if kind == "range" and minimum is None and maximum is None:
        raise UserError("SUPERVISION_RULE_VALUE_INVALID", "range 规则至少需要 --min 或 --max。")
    if kind == "batch-alignment":
        if mode == "penalty":
            raise UserError("SUPERVISION_RULE_VALUE_INVALID", "batch-alignment 是结构检查，不支持 penalty。")
        if not isinstance(target_path, str) or not _valid_batch_target_path(target_path):
            raise UserError("SUPERVISION_RULE_VALUE_INVALID", "batch-alignment 需要合法的 --target-path，例如 batch.args[1] 或 batch.kwargs.labels。")
        if type(axis) is not int or not 0 <= axis <= 16:
            raise UserError("SUPERVISION_RULE_VALUE_INVALID", "batch-alignment 的 --axis 必须是 0 到 16 的整数。")
        if minimum is not None or maximum is not None or weight is not None:
            raise UserError("SUPERVISION_RULE_VALUE_INVALID", "batch-alignment 不接受数值范围或 penalty 权重。")
    elif target_path is not None:
        raise UserError("SUPERVISION_RULE_VALUE_INVALID", "只有 batch-alignment 接受 --target-path。")
    elif axis is not None and (type(axis) is not int or not 0 <= axis <= 16):
        raise UserError("SUPERVISION_RULE_VALUE_INVALID", "数值规则的样本轴须是 0 到 16 的整数。")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise UserError("SUPERVISION_RULE_VALUE_INVALID", "range 的最小值不能大于最大值。")
    if mode == "penalty" and (weight is None or weight <= 0):
        raise UserError("SUPERVISION_RULE_VALUE_INVALID", "penalty 必须显式提供大于 0 的 weight。")
    if mode == "penalty" and kind not in {"range", "custom-penalty"}:
        raise UserError("SUPERVISION_RULE_VALUE_INVALID", "当前只有 range 和 custom-penalty 可预览 differentiable penalty。")
    if mode != "penalty" and weight is not None:
        raise UserError("SUPERVISION_RULE_VALUE_INVALID", "只有 penalty 模式接受 weight。")
    if kind in {"custom-validator", "custom-penalty"}:
        if not _valid_custom_plugin(plugin, kind, mode):
            raise UserError("SUPERVISION_PLUGIN_INVALID", "自定义规则需要项目内 module:callable、model-maven-supervision-v1 协议及 JSON 对象参数。")
        if minimum is not None or maximum is not None or target_path is not None:
            raise UserError("SUPERVISION_PLUGIN_INVALID", "自定义规则由 callable 定义行为，不接受 min/max/target。")
        if axis is not None and (kind != "custom-validator" or type(axis) is not int or not 0 <= axis <= 16):
            raise UserError("SUPERVISION_PLUGIN_INVALID", "只有 custom-validator 可指定 0 到 16 的样本轴。")
        if kind == "custom-penalty" and (weight is None or weight <= 0):
            raise UserError("SUPERVISION_RULE_VALUE_INVALID", "custom-penalty 必须显式提供大于 0 的 weight。")
        if kind == "custom-validator" and weight is not None:
            raise UserError("SUPERVISION_RULE_VALUE_INVALID", "custom-validator 不接受 penalty weight。")
    elif plugin is not None:
        raise UserError("SUPERVISION_PLUGIN_INVALID", "只有自定义规则接受 plugin。")
    rule_id = rule_id or uuid.uuid4().hex[:16]
    if not isinstance(rule_id, str) or not _MARK_ID.fullmatch(rule_id):
        raise UserError("SUPERVISION_RULE_ID_INVALID", "规则 ID 仅支持 1 到 80 位字母、数字、点、下划线或连字符。")
    root, path = _state_path(project_root)
    state = _read_state(path)
    mark = next((item for item in state["marks"] if item["id"] == mark_id), None)
    if mark is None:
        raise UserError("SEMANTIC_MARK_NOT_FOUND", f"找不到语义标记：{mark_id}。")
    if any(item["id"] == mark_id and item["binding_status"] == "stale" for item in list_marks(root)["marks"]):
        raise UserError("SEMANTIC_MARK_STALE", "不能给已过期的语义标记添加规则。", "先使用 model retag 重新确认实现块和端口。")
    if any(rule["id"] == rule_id for item in state["marks"] for rule in item.get("rules", [])):
        raise UserError("SUPERVISION_RULE_ID_EXISTS", f"监督规则 ID 已存在：{rule_id}。")
    if len(mark.get("rules", [])) >= _MAX_RULES_PER_MARK:
        raise UserError("SUPERVISION_RULE_LIMIT", f"每个语义标记最多保存 {_MAX_RULES_PER_MARK} 条规则。")
    rule = {
        "id": rule_id,
        "kind": kind,
        "mode": mode,
        "min": minimum,
        "max": maximum,
        "weight": weight,
        "target_path": target_path,
        "axis": axis,
        "plugin": plugin,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    mark.setdefault("rules", []).append(rule)
    _write_state(path, state)
    return {"mark_id": mark_id, "rule": rule, "path": path.relative_to(root).as_posix()}


def list_rules(project_root):
    root, path = _state_path(project_root)
    state = _read_state(path)
    rules = [
        {"mark_id": mark["id"], "mark_role": mark["role"], "module_path": mark["module_path"],
         "port": mark["port"], **rule}
        for mark in state["marks"] for rule in mark.get("rules", [])
    ]
    return {"rules": rules, "count": len(rules), "path": path.relative_to(root).as_posix(), "state_revision": _state_revision(state)}


@_serialized_mutation
def remove_rule(project_root, rule_id, expected_rule=None):
    root, path = _state_path(project_root)
    state = _read_state(path)
    for mark in state["marks"]:
        remaining = [rule for rule in mark.get("rules", []) if rule.get("id") != rule_id]
        if len(remaining) != len(mark.get("rules", [])):
            selected = next(rule for rule in mark["rules"] if rule.get("id") == rule_id)
            actual = {"mark_id": mark["id"], "mark_role": mark["role"], "module_path": mark["module_path"], "port": mark["port"], **selected}
            if expected_rule is not None and expected_rule != actual:
                raise UserError("SUPERVISION_RULE_CHANGED", "规则已变化，未删除；请刷新后重新确认。")
            mark["rules"] = remaining
            _write_state(path, state)
            return {"removed": rule_id, "mark_id": mark["id"], "path": path.relative_to(root).as_posix()}
    raise UserError("SUPERVISION_RULE_NOT_FOUND", f"找不到监督规则：{rule_id}。")
