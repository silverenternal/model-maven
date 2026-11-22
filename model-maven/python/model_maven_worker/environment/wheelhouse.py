"""Integrity checked, interpreter-specific wheel archives for offline setup."""

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from ..protocol import UserError
from .common import (
    ENVIRONMENT_LOCK_PATH,
    atomic_write_bytes,
    distribution_name_key,
    python_identity,
    run_environment_process,
    safe_output_path,
    safe_project_path,
)
from .lock import validate_environment_lock
from .setup import compare_environment


WHEELHOUSE_INDEX = "index.json"
WHEELHOUSE_SCHEMA = 1
_SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
_WHEEL_FILENAME = re.compile(
    r"^(?P<distribution>[A-Za-z0-9][A-Za-z0-9_.]*)-(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)-"
    r"(?P<python>[A-Za-z0-9_.]+)-(?P<abi>[A-Za-z0-9_.]+)-(?P<platform>[A-Za-z0-9_.]+)\.whl$"
)


def _bundle_path(relative=None, lock_sha256=None):
    if relative is None:
        if not isinstance(lock_sha256, str) or not _SHA256.fullmatch(lock_sha256):
            raise UserError("WHEELHOUSE_LOCK_INVALID", "依赖锁缺少有效的 SHA-256 指纹。")
        relative = f".model-maven/wheelhouse/{lock_sha256.lower()}"
    if not isinstance(relative, str) or not relative:
        raise UserError("WHEELHOUSE_PATH_INVALID", "wheel archive 必须是项目内相对目录。")
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UserError("WHEELHOUSE_PATH_INVALID", "wheel archive 必须是项目内相对目录。")
    # safe_output_path checks every existing parent for symlinks or non-directories.
    return safe_output_path((path / WHEELHOUSE_INDEX).as_posix(), "wheel archive").parent


def _read_lock():
    lock_path = safe_project_path(ENVIRONMENT_LOCK_PATH.as_posix(), "依赖锁", must_exist=True)
    metadata_path = safe_project_path(".model-maven/environment.lock.json", "依赖锁元数据", must_exist=True)
    try:
        lock_bytes = lock_path.read_bytes()
        metadata_bytes = metadata_path.read_bytes()
        if len(metadata_bytes) > 2 * 1024 * 1024:
            raise ValueError("lock metadata is too large")
        metadata = json.loads(metadata_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise UserError("ENVIRONMENT_LOCK_INVALID", "无法读取依赖锁或元数据。") from error
    validate_environment_lock(metadata, lock_bytes)
    return metadata, lock_bytes


def _wheel_identity(filename):
    if not isinstance(filename, str) or Path(filename).name != filename:
        return None
    match = _WHEEL_FILENAME.fullmatch(filename)
    if match is None:
        return None
    return (
        distribution_name_key(match.group("distribution")),
        match.group("version"),
    )


def _load_index(index_path):
    try:
        if not index_path.is_file() or index_path.is_symlink():
            return None
        if index_path.stat().st_size > 4 * 1024 * 1024:
            return None
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(index, dict)
        or index.get("schema_version") != WHEELHOUSE_SCHEMA
        or not isinstance(index.get("lock_sha256"), str)
        or not _SHA256.fullmatch(index["lock_sha256"])
        or not isinstance(index.get("python"), dict)
        or not isinstance(index.get("files"), list)
        or len(index["files"]) > 4096
        or index.get("source_kind") != "configured-pip-source"
    ):
        return None
    seen = set()
    for item in index["files"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("filename"), str)
            or Path(item["filename"]).name != item["filename"]
            or not _SHA256.fullmatch(str(item.get("sha256", "")))
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] < 0
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("version"), str)
            or item.get("source_kind") != "configured-pip-source"
        ):
            return None
        key = distribution_name_key(item["name"])
        identity = _wheel_identity(item["filename"])
        if key in seen or identity != (key, item["version"]):
            return None
        seen.add(key)
    return index


def inspect_wheel_archive(python, relative=None):
    """Check archive index and referenced wheel files as separate facts."""
    metadata, lock_bytes = _read_lock()
    bundle = _bundle_path(relative, metadata["lock_sha256"])
    index_path = bundle / WHEELHOUSE_INDEX
    index = _load_index(index_path)
    index_available = index is not None
    if not index_available:
        return {
            "index_available": False,
            "archive_available": False,
            "path": bundle.as_posix(),
            "missing": [],
            "corrupt": [],
            "compatible": False,
        }

    compatible = (
        index["lock_sha256"] == metadata["lock_sha256"]
        and index["python"] == metadata["python"]
        and python_identity(python) == metadata["python"]
    )
    expected = {
        distribution_name_key(item["name"]): item
        for item in metadata["packages"]
    }
    indexed = {
        distribution_name_key(item["name"]): item
        for item in index["files"]
    }
    missing = sorted(set(expected) - set(indexed))
    corrupt = []
    for key in sorted(set(expected) & set(indexed)):
        entry = indexed[key]
        package = expected[key]
        wheel_path = bundle / entry["filename"]
        try:
            if wheel_path.is_symlink() or not wheel_path.is_file():
                missing.append(entry["filename"])
                continue
            digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
            if (
                entry["name"] != package["name"]
                or entry["version"] != package["version"]
                or entry["sha256"].lower() != package["sha256"].lower()
                or digest != entry["sha256"].lower()
                or wheel_path.stat().st_size != entry["size_bytes"]
            ):
                corrupt.append(entry["filename"])
        except OSError:
            missing.append(entry["filename"])
    extra = sorted(set(indexed) - set(expected))
    corrupt.extend(indexed[key]["filename"] for key in extra)
    archive_available = compatible and not missing and not corrupt and len(indexed) == len(expected)
    return {
        "index_available": True,
        "archive_available": archive_available,
        "path": bundle.as_posix(),
        "missing": sorted(missing),
        "corrupt": sorted(set(corrupt)),
        "compatible": compatible,
    }


def archive_environment_wheels(python, relative=None, stream_output=False):
    """Download only the locked, hash-pinned wheels into an immutable bundle.

    Pip uses the user's existing index/find-links configuration. Only the
    source category is recorded; URLs, index credentials, and pip config are
    never copied into the archive.
    """
    metadata, lock_bytes = _read_lock()
    identity = python_identity(python)
    if identity != metadata["python"]:
        raise UserError(
            "WHEELHOUSE_ABI_MISMATCH",
            "当前 Python ABI 与依赖锁不一致，不能为该锁生成 wheel archive。",
            "使用生成依赖锁时的同一解释器/ABI；本 archive 不承诺跨 Python、平台、驱动或加速器移植。",
        )
    bundle = _bundle_path(relative, metadata["lock_sha256"])
    if bundle.exists():
        check = inspect_wheel_archive(python, bundle.relative_to(Path.cwd()).as_posix())
        if check["archive_available"]:
            return {**check, "created": False, "file_count": len(metadata["packages"])}
        raise UserError(
            "WHEELHOUSE_ALREADY_EXISTS",
            "目标 wheel archive 已存在但索引、ABI 或 wheel 文件校验未通过；未覆盖现有材料。",
            "检查 env wheelhouse check 的 missing/corrupt 结果，手动备份后选择新 archive 路径。",
            check,
        )

    lock_path = safe_project_path(ENVIRONMENT_LOCK_PATH.as_posix(), "依赖锁", must_exist=True)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{metadata['lock_sha256'][:12]}-", dir=bundle.parent))
    try:
        run_environment_process(
            [
                python, "-m", "pip", "download", "--no-cache-dir", "--disable-pip-version-check",
                "--require-hashes", "--only-binary=:all:", "--no-deps", "--dest", str(staging),
                "-r", str(lock_path),
            ],
            "锁定 wheel 归档",
            stream_output=stream_output,
        )
        expected = {
            distribution_name_key(item["name"]): item
            for item in metadata["packages"]
        }
        found = {}
        for wheel in sorted(staging.iterdir()):
            identity_from_name = _wheel_identity(wheel.name)
            if wheel.is_symlink() or not wheel.is_file() or identity_from_name is None:
                raise UserError("WHEELHOUSE_ARTIFACT_INVALID", f"pip 返回了无效 wheel 文件：{wheel.name}")
            key, version = identity_from_name
            package = expected.get(key)
            if package is None or package["version"] != version or key in found:
                raise UserError("WHEELHOUSE_ARTIFACT_INVALID", f"wheel 与依赖锁不匹配或重复：{wheel.name}")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            if digest != package["sha256"].lower():
                raise UserError("WHEELHOUSE_HASH_MISMATCH", f"wheel SHA-256 与依赖锁不匹配：{wheel.name}")
            found[key] = {
                "name": package["name"],
                "version": package["version"],
                "filename": wheel.name,
                "sha256": digest,
                "size_bytes": wheel.stat().st_size,
                "source_kind": "configured-pip-source",
            }
        missing = sorted(set(expected) - set(found))
        if missing:
            raise UserError(
                "WHEELHOUSE_INCOMPLETE",
                "pip 没有为全部锁定包提供 wheel，未发布不完整 archive。",
                "检查当前索引/wheelhouse 是否有适用于此 Python ABI 的所有 wheel。",
                {"missing": missing},
            )
        if lock_path.read_bytes() != lock_bytes:
            raise UserError("ENVIRONMENT_LOCK_CHANGED", "wheel 归档期间依赖锁发生变化，未发布 archive。")
        index = {
            "schema_version": WHEELHOUSE_SCHEMA,
            "lock_sha256": metadata["lock_sha256"],
            "python": identity,
            "source_kind": "configured-pip-source",
            "files": sorted(found.values(), key=lambda item: distribution_name_key(item["name"])),
        }
        atomic_write_bytes(staging / WHEELHOUSE_INDEX, (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode())
        os.replace(staging, bundle)
        check = inspect_wheel_archive(python, bundle.relative_to(Path.cwd()).as_posix())
        if not check["archive_available"]:
            raise UserError("WHEELHOUSE_VERIFY_FAILED", "wheel archive 写入后未通过索引/文件校验。", data=check)
        return {**check, "created": True, "file_count": len(found)}
    except FileExistsError as error:
        raise UserError("WHEELHOUSE_ALREADY_EXISTS", "wheel archive 目标被并发创建，未覆盖现有材料。") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def replay_wheel_archive(python, venv, relative=None, stream_output=False):
    """Install the locked dependency closure without network access."""
    metadata, lock_bytes = _read_lock()
    check = inspect_wheel_archive(python, relative)
    if not check["archive_available"]:
        raise UserError(
            "WHEELHOUSE_INCOMPLETE",
            "wheel archive 不完整、损坏或不适用于当前 Python ABI；未开始安装。",
            "运行 env wheelhouse check 获取 missing/corrupt/compatible 详情，联网重新归档缺失 wheel。",
            check,
        )
    target = safe_project_path(venv, "虚拟环境")
    target_python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if target.is_symlink() or not target.is_dir() or not target_python.is_file():
        raise UserError("ENVIRONMENT_TARGET_INVALID", "离线安装目标必须是项目内已存在的虚拟环境。")
    identity = python_identity(target_python.as_posix())
    if identity != metadata["python"]:
        raise UserError(
            "WHEELHOUSE_ABI_MISMATCH",
            "目标虚拟环境 Python ABI 与 wheel archive 不一致，未安装依赖。",
            "为生成 archive 的同一 Python ABI 创建新虚拟环境；本材料不覆盖 OS、驱动或加速器兼容性。",
        )
    bundle = _bundle_path(relative, metadata["lock_sha256"])
    lock_path = safe_project_path(ENVIRONMENT_LOCK_PATH.as_posix(), "依赖锁", must_exist=True)
    # Keep the lock bytes stable across the external pip invocation.
    if hashlib.sha256(lock_path.read_bytes()).digest() != hashlib.sha256(lock_bytes).digest():
        raise UserError("ENVIRONMENT_LOCK_CHANGED", "离线安装期间依赖锁发生变化，未继续安装。")
    run_environment_process(
        [
            target_python.as_posix(), "-m", "pip", "--isolated", "install", "--no-index", "--find-links", str(bundle),
            "--require-hashes", "--only-binary=:all:", "--no-deps", "--disable-pip-version-check", "-r", str(lock_path),
        ],
        "离线锁定依赖安装",
        stream_output=stream_output,
    )
    environment_check = compare_environment(target_python.as_posix(), metadata, metadata["requirements_file"])
    if not environment_check["matches"]:
        raise UserError(
            "ENVIRONMENT_VERIFY_FAILED",
            "离线依赖安装后环境与锁不一致。",
            "查看缺失/版本差异后修复 wheel archive；该次操作不会标记为成功。",
            environment_check,
        )
    return {
        "venv": venv,
        "environment_check": environment_check,
        "wheelhouse": check["path"],
        "offline": True,
        "note": "已从同 ABI wheel archive 离线安装并通过依赖锁与 pip check。",
    }
