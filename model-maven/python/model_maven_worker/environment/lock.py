"""Recursive requirement input validation and wheel-hash lock resolution."""

import hashlib
import json
import os
import re
import os
import subprocess
import tempfile
from pathlib import Path

from ..protocol import UserError
from .common import (
    ENVIRONMENT_LOCK_PATH,
    ENVIRONMENT_LOCK_SCHEMA,
    ENVIRONMENT_METADATA_PATH,
    atomic_write_bytes,
    distribution_name_key,
    python_identity,
    run_environment_process,
    safe_output_path,
    safe_project_path,
    digest,
    redact_environment_output,
)

def lock_file_paths():
    return (
        safe_output_path(ENVIRONMENT_LOCK_PATH.as_posix(), "依赖锁"),
        safe_output_path(ENVIRONMENT_METADATA_PATH.as_posix(), "依赖锁元数据"),
    )


def _system_site_packages_inventory(python):
    script = (
        "import importlib.metadata,json; "
        "print(json.dumps(sorted([{'name':d.metadata.get('Name',''),'version':d.version} "
        "for d in importlib.metadata.distributions() if d.metadata.get('Name') and d.version], "
        "key=lambda p:p['name'].lower())))"
    )
    result = run_environment_process([python, "-c", script], "基础环境包清单读取", timeout=60)
    try:
        packages = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise UserError("ENVIRONMENT_INVENTORY_INVALID", "基础 Python 没有返回有效的包清单。") from error
    unique = {}
    for item in packages:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("version"), str):
            raise UserError("ENVIRONMENT_INVENTORY_INVALID", "基础 Python 包清单包含无效条目。")
        key = distribution_name_key(item["name"])
        if key in {"pip", "setuptools", "wheel"}:
            continue
        # importlib.metadata can see duplicate distributions on layered Python
        # installs. Match installed_environment() semantics: later entries in
        # its stable, name-sorted inventory are the effective value.
        unique[key] = item["version"]
    distributions = [{"name": name, "version": version} for name, version in sorted(unique.items())]
    environment = os.environ.copy()
    environment.update({"PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        checked = subprocess.run(
            [python, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise UserError("ENVIRONMENT_BASE_CHECK_FAILED", f"无法读取基础 Python 的 pip check：{error}") from error
    findings = sorted(
        line for line in (checked.stdout + "\n" + checked.stderr).splitlines()
        if line.strip() and line.strip() != "No broken requirements found."
    )
    findings = [redact_environment_output(line, limit=2000) for line in findings]
    if len(findings) > 10000 or sum(map(len, findings)) > 1024 * 1024:
        raise UserError("ENVIRONMENT_BASE_CHECK_TOO_LARGE", "基础 Python 的 pip check 输出超过允许上限。")
    pip_check = {"findings": findings, "fingerprint": digest(findings)}
    return {
        "enabled": True,
        "distributions": distributions,
        "fingerprint": digest(distributions),
        "pip_check": pip_check,
    }


def validate_environment_lock(metadata, lock_bytes):
    if not isinstance(metadata, dict) or metadata.get("schema_version") != ENVIRONMENT_LOCK_SCHEMA:
        raise UserError(
            "ENVIRONMENT_LOCK_INVALID",
            "依赖锁元数据版本或格式无效。",
            "保留现有环境材料，检查 requirements 输入后运行 `m-model env lock --refresh` 生成当前版本锁。",
        )
    if not isinstance(lock_bytes, bytes) or len(lock_bytes) > 2 * 1024 * 1024:
        raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁文件超过 2 MiB 或格式无效。")
    if metadata.get("lock_sha256") != hashlib.sha256(lock_bytes).hexdigest():
        raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁与元数据校验失败。")
    if not isinstance(metadata.get("requirements_file"), str) or not re.fullmatch(
        r"[a-fA-F0-9]{64}", str(metadata.get("requirements_sha256", ""))
    ):
        raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁缺少有效的 requirements 输入指纹。")
    identity = metadata.get("python")
    identity_fields = {"implementation", "version", "cache_tag", "platform", "machine", "soabi"}
    if not isinstance(identity, dict) or not identity_fields.issubset(identity):
        raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁缺少有效的 Python ABI 信息。")
    packages = metadata.get("packages")
    if not isinstance(packages, list) or not packages or len(packages) > 4096:
        raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁没有有效的包清单。")
    seen = set()
    for item in packages:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", item["name"])
            or not isinstance(item.get("version"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,199}", item["version"])
            or not isinstance(item.get("sha256"), str)
            or not re.fullmatch(r"[a-fA-F0-9]{64}", item["sha256"])
        ):
            raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁包含无效的包版本或 artifact 哈希。")
        normalized = distribution_name_key(item["name"])
        if normalized in seen:
            raise UserError("ENVIRONMENT_LOCK_INVALID", f"依赖锁中存在重复包：{item['name']}")
        seen.add(normalized)
    inherited = metadata.get("system_site_packages", {
        "enabled": False,
        "distributions": [],
        "fingerprint": digest([]),
        "pip_check": {"findings": [], "fingerprint": digest([])},
    })
    if not isinstance(inherited, dict) or type(inherited.get("enabled")) is not bool:
        raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁的基础环境继承策略无效。")
    inherited_distributions = inherited.get("distributions")
    if not isinstance(inherited_distributions, list) or len(inherited_distributions) > 10000:
        raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁的继承包清单无效。")
    inherited_seen = set()
    normalized_distributions = []
    for item in inherited_distributions:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", item["name"])
            or not isinstance(item.get("version"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,199}", item["version"])
        ):
            raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁的继承包条目无效。")
        key = distribution_name_key(item["name"])
        if key in inherited_seen:
            raise UserError("ENVIRONMENT_LOCK_INVALID", f"继承包清单存在重复包：{item['name']}")
        inherited_seen.add(key)
        normalized_distributions.append({"name": item["name"], "version": item["version"]})
    expected_fingerprint = digest(normalized_distributions)
    if (not inherited.get("enabled") and inherited_distributions) or inherited.get("fingerprint") != expected_fingerprint:
        raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁的继承包指纹校验失败。")
    pip_check = inherited.get("pip_check", {"findings": [], "fingerprint": digest([])})
    if (
        not isinstance(pip_check, dict)
        or not isinstance(pip_check.get("findings"), list)
        or any(not isinstance(item, str) for item in pip_check.get("findings", []))
        or pip_check.get("fingerprint") != digest(pip_check.get("findings", []))
    ):
        raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁的基础 pip check 基线无效。")
    return metadata


def load_environment_lock(requirements_digest, identity, refresh, system_site_packages=None):
    text_path, metadata_path = lock_file_paths()
    if not text_path.exists() and not metadata_path.exists():
        return None
    if not text_path.exists() or not metadata_path.exists():
        if not refresh:
            raise UserError(
                "ENVIRONMENT_LOCK_INCOMPLETE",
                "依赖锁文件与元数据不完整，未自动覆盖现有文件。",
                "检查 requirements.lock.txt 和 .model-maven/environment.lock.json；确认可覆盖后运行 env lock --refresh。",
            )
        return None
    try:
        text = text_path.read_bytes()
        if len(text) > 2 * 1024 * 1024:
            raise ValueError("lock file is too large")
        metadata_bytes = metadata_path.read_bytes()
        if len(metadata_bytes) > 2 * 1024 * 1024:
            raise ValueError("lock metadata is too large")
        metadata = json.loads(metadata_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if not refresh:
            raise UserError("ENVIRONMENT_LOCK_INVALID", "依赖锁文件或元数据无法读取；未自动覆盖。", "修复文件或确认后运行 env lock --refresh。") from error
        return None
    try:
        validate_environment_lock(metadata, text)
    except UserError:
        if not refresh:
            raise
        return None
    if metadata.get("requirements_sha256") != requirements_digest or metadata.get("python") != identity:
        if not refresh:
            raise UserError(
                "ENVIRONMENT_LOCK_STALE",
                "依赖输入或 Python ABI 与现有锁不同；不会静默改写锁。",
                "检查 requirements.txt 与目标 Python；确认更新后运行 env lock --refresh 或 env setup --update-lock。",
            )
        return None
    packages = metadata["packages"]
    actual_inheritance = metadata.get("system_site_packages", {
        "enabled": False,
        "distributions": [],
        "fingerprint": digest([]),
        "pip_check": {"findings": [], "fingerprint": digest([])},
    })
    if "pip_check" not in actual_inheritance:
        actual_inheritance = {**actual_inheritance, "pip_check": {"findings": [], "fingerprint": digest([])}}
    expected_inheritance = system_site_packages or {
        "enabled": False,
        "distributions": [],
        "fingerprint": digest([]),
        "pip_check": {"findings": [], "fingerprint": digest([])},
    }
    if actual_inheritance != expected_inheritance:
        if not refresh:
            raise UserError("ENVIRONMENT_LOCK_STALE", "项目基础环境继承策略或包指纹与现有锁不同；不会静默改写锁。", "检查基础 Python 与继承策略；确认后运行 env lock --refresh。")
        return None
    return {"metadata": metadata, "text": text.decode("utf-8"), "packages": packages, "cached": True}


def requirements_inputs_digest(relative, visited=None, depth=0, active=None):
    if visited is None:
        visited = {}
    if active is None:
        active = set()
    if depth > 16:
        raise UserError("ENVIRONMENT_INPUT_TOO_DEEP", "requirements 文件递归包含超过 16 层。")
    path = safe_project_path(relative, "依赖输入", must_exist=True)
    project_relative = path.relative_to(Path.cwd().resolve(strict=True)).as_posix()
    if project_relative in active:
        raise UserError("ENVIRONMENT_INPUT_CYCLE", f"requirements 文件存在循环包含：{project_relative}")
    if project_relative in visited:
        return visited
    try:
        contents = path.read_bytes()
        decoded = contents.decode("utf-8")
    except OSError as error:
        raise UserError("ENVIRONMENT_INPUT_INVALID", f"无法读取依赖输入：{project_relative}") from error
    except UnicodeDecodeError as error:
        raise UserError("ENVIRONMENT_INPUT_INVALID", f"依赖输入必须是 UTF-8 文本：{project_relative}") from error
    if len(contents) > 1024 * 1024:
        raise UserError("ENVIRONMENT_INPUT_INVALID", f"依赖输入超过 1 MiB：{project_relative}")
    visited[project_relative] = contents
    active.add(project_relative)
    for raw_line in decoded.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        lowered = line.lower()
        option = lowered.split("=", 1)[0].split(None, 1)[0]
        include = re.fullmatch(r"(?:-r|--requirement|-c|--constraint)(?:\s+|=)(\S+)", line)
        if include:
            include_path = Path(include.group(1))
            if include_path.is_absolute() or any(part in {"", ".", ".."} for part in include_path.parts):
                raise UserError("ENVIRONMENT_INPUT_UNSAFE", f"依赖 include 必须留在项目目录内：{include.group(1)}")
            included = Path(project_relative).parent / include_path
            requirements_inputs_digest(included.as_posix(), visited, depth + 1, active)
            continue
        if option in {"--index-url", "--extra-index-url", "--find-links", "--no-index", "--trusted-host", "-i", "-f"}:
            raise UserError(
                "ENVIRONMENT_SOURCE_CONFIGURATION_UNSUPPORTED",
                "requirements 文件中的包索引/查找目录选项不会被复制进可移植锁。",
                "把私有索引或 wheelhouse 配置到 pip config、PIP_INDEX_URL 或 PIP_FIND_LINKS，使解析与安装使用同一来源。",
            )
        if option in {"-e", "--editable"} or lowered.startswith(("git+", "hg+", "svn+", "bzr+", "--no-binary")):
            raise UserError(
                "ENVIRONMENT_SOURCE_DEPENDENCY_UNSUPPORTED",
                "当前环境锁不接受 editable、VCS 或源码构建输入。",
                "将组件打包为 wheel 并通过包索引/--find-links 提供；源码构建不会在解析阶段执行。",
            )
        direct_reference = re.search(r"\s@\s*(\S+)", line)
        direct_url = direct_reference.group(1) if direct_reference else line
        if direct_reference or direct_url.lower().startswith(("http://", "https://", "file:")):
            raise UserError(
                "ENVIRONMENT_SOURCE_DEPENDENCY_UNSUPPORTED",
                "环境锁不保留直接 URL 或本地 wheel 路径，因此拒绝生成无法复用的锁。",
                "将 wheel 发布到当前 pip 配置可访问的索引，或通过 pip config/PIP_FIND_LINKS 配置 wheelhouse。",
            )
        if (
            lowered.startswith(("./", "../", ".\\", "..\\", "/", "~"))
            or re.match(r"^[a-z]:[\\/]", lowered)
            or line.startswith("\\\\")
        ):
            raise UserError(
                "ENVIRONMENT_SOURCE_DEPENDENCY_UNSUPPORTED",
                "当前环境锁不接受本地源码路径。",
                "将依赖构建为 wheel 并通过包索引或 --find-links 提供。",
            )
    active.remove(project_relative)
    return visited


def requirements_content_digest(relative):
    requirement_files = requirements_inputs_digest(relative)
    digest_value = hashlib.sha256()
    total_input_bytes = 0
    for path, contents in sorted(requirement_files.items()):
        total_input_bytes += len(contents)
        if total_input_bytes > 1024 * 1024:
            raise UserError("ENVIRONMENT_INPUT_INVALID", "requirements 文件及递归输入合计超过 1 MiB。")
        digest_value.update(path.encode("utf-8"))
        digest_value.update(b"\0")
        digest_value.update(contents)
        digest_value.update(b"\0")
    return digest_value.hexdigest(), requirement_files


def resolve_environment_lock(python, requirements, refresh=False, stream_output=False, system_site_packages=False):
    requirements_path = safe_project_path(requirements, "依赖输入", must_exist=True)
    requirements_digest, requirement_files = requirements_content_digest(requirements)
    identity = python_identity(python)
    inheritance = _system_site_packages_inventory(python) if system_site_packages else {
        "enabled": False,
        "distributions": [],
        "fingerprint": digest([]),
        "pip_check": {"findings": [], "fingerprint": digest([])},
    }
    cached = load_environment_lock(requirements_digest, identity, refresh, inheritance)
    if cached is not None:
        return cached

    lock_text_path, metadata_path = lock_file_paths()
    report_file = tempfile.NamedTemporaryFile(prefix="model-maven-pip-report-", suffix=".json", delete=False)
    report_path = Path(report_file.name)
    report_file.close()
    try:
        run_environment_process(
            [
                python, "-m", "pip", "install", "--dry-run", "--ignore-installed",
                "--report", str(report_path), "--only-binary=:all:", "--disable-pip-version-check",
                "-r", str(requirements_path),
            ],
            "依赖解析",
            stream_output=stream_output,
        )
        try:
            report_bytes = report_path.read_bytes()
            if len(report_bytes) > 16 * 1024 * 1024:
                raise ValueError("report is too large")
            report = json.loads(report_bytes)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise UserError("ENVIRONMENT_RESOLUTION_INVALID", "pip 没有生成可读取的依赖报告。") from error
    finally:
        try:
            report_path.unlink()
        except FileNotFoundError:
            pass

    installs = report.get("install") if isinstance(report, dict) else None
    if not isinstance(installs, list) or not installs:
        raise UserError("ENVIRONMENT_RESOLUTION_EMPTY", "依赖输入没有解析出可安装的 wheel 包。", "在 requirements.txt 声明项目实际需要的运行依赖后重试。")
    packages = []
    seen = set()
    for item in installs:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        download = item.get("download_info") if isinstance(item, dict) else None
        archive = download.get("archive_info") if isinstance(download, dict) else None
        name = metadata.get("name") if isinstance(metadata, dict) else None
        version = metadata.get("version") if isinstance(metadata, dict) else None
        hashes = archive.get("hashes", {}) if isinstance(archive, dict) else {}
        sha256 = hashes.get("sha256") if isinstance(hashes, dict) else None
        if not sha256 and isinstance(archive, dict):
            legacy_hash = archive.get("hash")
            if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
                sha256 = legacy_hash[7:]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", name)
            or not isinstance(version, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,199}", version)
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[a-fA-F0-9]{64}", sha256)
        ):
            raise UserError(
                "ENVIRONMENT_ARTIFACT_UNHASHED",
                "解析结果包含没有 SHA-256 wheel 哈希的包，未生成不完整锁。",
                "检查包索引是否提供可用 wheel；Model-Maven 当前只接受带 SHA-256 的 wheel。",
            )
        normalized = distribution_name_key(name)
        if normalized in seen:
            raise UserError("ENVIRONMENT_RESOLUTION_INVALID", f"解析结果中包名重复：{name}")
        seen.add(normalized)
        packages.append({"name": name, "version": version, "sha256": sha256.lower()})
    packages.sort(key=lambda item: distribution_name_key(item["name"]))
    lines = [
        "# Generated by Model-Maven. Review requirements.txt for top-level intent.",
        "# Resolved wheel artifacts are pinned by version and SHA-256.",
        "--require-hashes",
    ]
    for package in packages:
        lines.append(f"{package['name']}=={package['version']} --hash=sha256:{package['sha256']}")
    lock_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    metadata = {
        "schema_version": ENVIRONMENT_LOCK_SCHEMA,
        "requirements_file": requirements_path.relative_to(Path.cwd().resolve(strict=True)).as_posix(),
        "requirements_sha256": requirements_digest,
        "requirements_inputs": sorted(requirement_files),
        "python": identity,
        "packages": packages,
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "resolver": "pip dry-run, wheel-only",
        "system_site_packages": inheritance,
    }
    # Validate output paths again immediately before replacing files.
    lock_text_path, metadata_path = lock_file_paths()
    atomic_write_bytes(lock_text_path, lock_bytes)
    atomic_write_bytes(metadata_path, (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return {"metadata": metadata, "text": lock_bytes.decode("utf-8"), "packages": packages, "cached": False}

