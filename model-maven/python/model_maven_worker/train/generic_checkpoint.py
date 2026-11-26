"""Framework-neutral JSON checkpoint primitives for explicit user state."""

import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..protocol import UserError
from .generic_loop import _json_state

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


def _identity(value):
    base = {
        "loop", "python_executable", "python_version", "source_manifest_sha256",
        "environment_fingerprint_sha256", "parameters_sha256", "runtime_paths_sha256",
    }
    if not isinstance(value, dict) or (set(value) != base and set(value) != base | {"loop_interface"}):
        raise UserError("GENERIC_CHECKPOINT_IDENTITY_INVALID", "纯 Python 循环恢复需要完整的运行身份。")
    if "loop_interface" in value and value["loop_interface"] != "context":
        raise UserError("GENERIC_CHECKPOINT_IDENTITY_INVALID", "纯 Python 循环签名身份无效。")
    for field in ("loop", "python_executable", "python_version"):
        if not isinstance(value[field], str) or not value[field] or len(value[field]) > 4096:
            raise UserError("GENERIC_CHECKPOINT_IDENTITY_INVALID", f"运行身份 {field} 无效。")
    for field in ("source_manifest_sha256", "environment_fingerprint_sha256", "parameters_sha256", "runtime_paths_sha256"):
        if not isinstance(value[field], str) or not _SHA256.fullmatch(value[field]):
            raise UserError("GENERIC_CHECKPOINT_IDENTITY_INVALID", f"运行身份 {field} 缺少有效摘要。")
    return dict(value)


def _encode(value):
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise UserError("GENERIC_CHECKPOINT_INVALID", "纯 Python checkpoint 必须是有限 JSON。") from error
    if len(raw) > _MAX_DOCUMENT_BYTES:
        raise UserError("GENERIC_CHECKPOINT_INVALID", "纯 Python checkpoint 超出大小上限。")
    return raw


def _read_regular(path):
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("not a regular file")
            raw = stream.read(_MAX_DOCUMENT_BYTES + 1)
        if len(raw) > _MAX_DOCUMENT_BYTES:
            raise ValueError("file too large")
        return raw
    except (OSError, ValueError) as error:
        raise UserError("GENERIC_CHECKPOINT_UNAVAILABLE", f"无法读取纯 Python checkpoint：{error}") from error


def _write_atomic(directory, destination, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".generic-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        if os.name == "posix":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError as error:
        raise UserError("GENERIC_CHECKPOINT_WRITE_FAILED", f"无法保存纯 Python checkpoint：{error}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


@contextmanager
def _checkpoint_lock(directory):
    """Serialize cooperating writers/readers; callers must not bypass this lock."""
    lock_path = directory / "generic.lock"
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as error:
        raise UserError("GENERIC_CHECKPOINT_LOCK_UNAVAILABLE", str(error)) from error
    locked = False
    try:
        if lock_path.is_symlink() or not stat.S_ISREG(os.fstat(fd).st_mode):
            raise UserError("GENERIC_CHECKPOINT_PATH_INVALID", "checkpoint 锁必须是普通文件。")
        if os.name == "nt":
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        yield
    except OSError as error:
        raise UserError("GENERIC_CHECKPOINT_LOCK_UNAVAILABLE", str(error)) from error
    finally:
        if locked:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def save_generic_checkpoint(directory, *, completed_steps, state, identity, expected_previous):
    """Publish under a lock only if the selected previous checkpoint matches.

    expected_previous=None explicitly requires no previous checkpoint; a
    digest requires that exact verified checkpoint and increasing step count.
    """
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise UserError("GENERIC_CHECKPOINT_PATH_INVALID", "checkpoint 目录必须是现有普通目录。")
    if type(completed_steps) is not int or completed_steps < 1:
        raise UserError("GENERIC_CHECKPOINT_INVALID", "checkpoint 步数必须为正整数。")
    checked_identity = _identity(identity)
    payload = {"schema_version": 1, "format": "model-maven-generic-json-v1",
               "completed_steps": completed_steps, "identity": checked_identity,
               "state": _json_state(state)}
    raw = _encode(payload)
    with _checkpoint_lock(directory):
        latest = directory / "latest.generic.json"
        if expected_previous is None:
            if os.path.lexists(latest):
                raise UserError("GENERIC_CHECKPOINT_CONFLICT", "恢复点已存在；不会覆盖未知状态。")
        else:
            prior = _load_generic_checkpoint(directory, expected_identity=checked_identity,
                                             expected_sha256=expected_previous)
            if completed_steps <= prior["completed_steps"]:
                raise UserError("GENERIC_CHECKPOINT_CONFLICT", "新恢复点步数必须大于已选择的恢复点。")
        name = f"generic-{completed_steps:09d}-{uuid.uuid4().hex}.json"
        target = directory / name
        _write_atomic(directory, target, raw)
        manifest = {"schema_version": 1, "format": "model-maven-generic-json-v1", "file": name,
                    "sha256": hashlib.sha256(raw).hexdigest(), "completed_steps": completed_steps}
        _write_atomic(directory, latest, _encode(manifest))
        return manifest


def load_generic_checkpoint(directory, *, expected_identity, expected_sha256):
    """Verify exact selected payload and identity; never infer compatible state."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise UserError("GENERIC_CHECKPOINT_PATH_INVALID", "checkpoint 目录必须是现有普通目录。")
    with _checkpoint_lock(directory):
        return _load_generic_checkpoint(directory, expected_identity=expected_identity,
                                        expected_sha256=expected_sha256)


def _load_generic_checkpoint(directory, *, expected_identity, expected_sha256):
    identity = _identity(expected_identity)
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise UserError("GENERIC_CHECKPOINT_IDENTITY_INVALID", "恢复必须指定 checkpoint SHA-256。")
    try:
        manifest = json.loads(_read_regular(directory / "latest.generic.json"))
        name = manifest["file"]
        if (manifest.get("schema_version") != 1 or manifest.get("format") != "model-maven-generic-json-v1"
                or not isinstance(name, str) or not re.fullmatch(r"generic-[0-9]{9,}-[0-9a-f]{32}\.json", name)
                or manifest.get("sha256") != expected_sha256):
            raise ValueError("invalid or changed manifest")
        raw = _read_regular(directory / name)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("checkpoint digest mismatch")
        payload = json.loads(raw)
        if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                or payload.get("format") != "model-maven-generic-json-v1"
                or type(payload.get("completed_steps")) is not int or payload["completed_steps"] < 1
                or payload["completed_steps"] != manifest.get("completed_steps")
                or payload.get("identity") != identity or "state" not in payload):
            raise ValueError("checkpoint identity or state mismatch")
        payload["state"] = _json_state(payload["state"])
        return payload
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise UserError("GENERIC_CHECKPOINT_CONFLICT", f"纯 Python checkpoint 无法验证：{error}",
                        "恢复原运行身份及所选恢复点，或开始新运行；没有执行用户循环。") from error
