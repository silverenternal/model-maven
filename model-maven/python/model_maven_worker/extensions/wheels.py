"""Safe wheel metadata inspection and project-environment dependency checks."""

import hashlib
import importlib.metadata
import json
import stat
import sys
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from ..protocol import UserError


@dataclass(frozen=True)
class ComponentWheelRuntime:
    distribution_name_key: Callable[[str], str]
    lock_file_paths: Callable[[], tuple[Path, Path]]
    validate_environment_lock: Callable[[dict, bytes], Any]
    python_identity: Callable[[str], dict]
    requirements_content_digest: Callable[[str], tuple[str, list]]
    python_executable: str


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def wheel_runtime_requirements(runtime, wheel_path):
    if not wheel_path.is_file() or wheel_path.is_symlink():
        raise UserError("COMPONENT_WHEEL_INVALID", "构建产物不是普通 wheel 文件。")
    if wheel_path.stat().st_size > 512 * 1024 * 1024:
        raise UserError("COMPONENT_WHEEL_TOO_LARGE", "扩展 wheel 超过 512 MiB。")
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            infos = archive.infolist()
            if len(infos) > 20000 or sum(info.file_size for info in infos) > 2 * 1024 * 1024 * 1024:
                raise UserError("COMPONENT_WHEEL_TOO_LARGE", "扩展 wheel 超过 20000 个文件或 2 GiB 解压上限。")
            metadata_names = []
            for info in infos:
                member = PurePosixPath(info.filename)
                if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
                    raise UserError("COMPONENT_WHEEL_INVALID", "wheel 包含不安全的归档路径。")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise UserError("COMPONENT_WHEEL_INVALID", "wheel 包含符号链接文件。")
                if len(member.parts) == 2 and member.parts[0].endswith(".dist-info") and member.name == "METADATA":
                    metadata_names.append(info.filename)
            if len(metadata_names) != 1:
                raise UserError("COMPONENT_WHEEL_INVALID", "wheel 必须包含唯一 distribution METADATA。")
            metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
            if not metadata.get("Name") or not metadata.get("Version"):
                raise UserError("COMPONENT_WHEEL_INVALID", "wheel METADATA 缺少 Name 或 Version。")
            raw_requirements = metadata.get_all("Requires-Dist", [])
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise UserError("COMPONENT_WHEEL_INVALID", f"无法读取构建 wheel 的 METADATA：{error}") from error

    try:
        from packaging.markers import default_environment
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        try:
            from pip._vendor.packaging.markers import default_environment
            from pip._vendor.packaging.requirements import Requirement
            from pip._vendor.packaging.version import Version
        except ImportError as error:
            raise UserError("COMPONENT_DEPENDENCY_CHECK_UNAVAILABLE", "当前 Python 缺少 PEP 508 依赖检查器。", "使用带 pip 的 Python 环境后重试。") from error

    locked_versions = None
    lock_path, metadata_path = runtime.lock_file_paths()
    if lock_path.exists() or metadata_path.exists():
        if not lock_path.is_file() or not metadata_path.is_file():
            raise UserError("ENVIRONMENT_LOCK_INCOMPLETE", "项目依赖锁文件或元数据不完整，扩展安装已停止。", "先修复 requirements.lock.txt 与 .model-maven/environment.lock.json，再运行 m-model env check。")
        try:
            lock_bytes = lock_path.read_bytes()
            if metadata_path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("lock metadata is too large")
            lock_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise UserError("ENVIRONMENT_LOCK_INVALID", "无法读取项目依赖锁；扩展安装已停止。") from error
        runtime.validate_environment_lock(lock_metadata, lock_bytes)
        if lock_metadata["python"] != runtime.python_identity(runtime.python_executable):
            raise UserError("ENVIRONMENT_LOCK_STALE", "当前所选 Python ABI 与项目依赖锁不同。", "使用 env setup 选择锁定环境，或先为当前环境生成新锁。")
        requirements_digest, _ = runtime.requirements_content_digest(lock_metadata["requirements_file"])
        if requirements_digest != lock_metadata["requirements_sha256"]:
            raise UserError("ENVIRONMENT_LOCK_STALE", "requirements 输入已变化，扩展运行依赖锁定状态不再有效。", "检查依赖输入并运行 m-model env lock --refresh 与 env setup。")
        locked_versions = {
            runtime.distribution_name_key(item["name"]): item["version"]
            for item in lock_metadata["packages"]
        }

    installed = {}
    duplicates = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "").strip()
        if not name:
            continue
        key = runtime.distribution_name_key(name)
        if key in installed:
            duplicates.add(key)
        installed[key] = distribution.version
    missing = []
    environment = default_environment()
    for raw in raw_requirements:
        try:
            requirement = Requirement(raw)
        except Exception as error:
            raise UserError("COMPONENT_WHEEL_INVALID", f"wheel 含无效的 PEP 508 依赖声明：{raw}") from error
        if requirement.url is not None:
            raise UserError(
                "COMPONENT_DEPENDENCY_UNSUPPORTED",
                f"扩展依赖 {raw} 使用直接 URL，无法由项目哈希锁稳定管理。",
                "将依赖发布为带版本约束的 distribution，并加入项目 requirements 锁。",
            )
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        key = runtime.distribution_name_key(requirement.name)
        version = installed.get(key)
        satisfied = version is not None and key not in duplicates
        locked_version = locked_versions.get(key) if locked_versions is not None else None
        if locked_versions is not None and (locked_version is None or version != locked_version):
            satisfied = False
        if satisfied and requirement.specifier:
            try:
                satisfied = requirement.specifier.contains(Version(version), prereleases=True)
                if locked_version is not None:
                    satisfied = satisfied and requirement.specifier.contains(
                        Version(locked_version), prereleases=True
                    )
            except Exception:
                satisfied = False
        if not satisfied:
            missing.append({"requirement": raw, "installed_version": version, "locked_version": locked_version, "ambiguous": key in duplicates})
    if missing:
        raise UserError(
            "COMPONENT_DEPENDENCY_MISSING",
            "当前项目 Python 环境不满足扩展声明的依赖；扩展安装不会改写环境或 requirements 锁。",
            "把缺失依赖加入项目 requirements.txt，运行 m-model env setup，再重新安装扩展。",
            {"missing": missing},
        )
    return {"name": metadata.get("Name"), "version": metadata.get("Version"), "requirements": raw_requirements}


