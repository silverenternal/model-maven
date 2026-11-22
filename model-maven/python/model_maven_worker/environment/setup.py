"""Project virtual environment preparation, selection, and drift checks."""

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

from ..protocol import UserError
from .common import (
    ENVIRONMENT_CONFIG_PATH,
    ENVIRONMENT_LOCK_PATH,
    ENVIRONMENT_METADATA_PATH,
    atomic_write_bytes,
    distribution_name_key,
    python_identity,
    redact_environment_output,
    run_environment_process,
    safe_output_path,
    safe_project_path,
)
from .lock import (
    lock_file_paths,
    requirements_content_digest,
    resolve_environment_lock,
    validate_environment_lock,
)

def _environment_config():
    config_path = safe_output_path(ENVIRONMENT_CONFIG_PATH.as_posix(), "项目环境配置")
    if not config_path.exists():
        return {"schema_version": 1, "venv": ".venv", "system_site_packages": False}
    try:
        if config_path.stat().st_size > 16 * 1024:
            raise ValueError("environment config is too large")
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise UserError("ENVIRONMENT_CONFIG_INVALID", "项目虚拟环境选择配置无法读取。", "检查 .model-maven/environment.json 后重试。") from error
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or not isinstance(config.get("venv", ".venv"), str)
        or type(config.get("system_site_packages", False)) is not bool
    ):
        raise UserError("ENVIRONMENT_CONFIG_INVALID", "项目虚拟环境选择配置格式无效。", "检查 .model-maven/environment.json 中的 venv/system_site_packages 字段。")
    return {"schema_version": 1, "venv": config.get("venv", ".venv"), "system_site_packages": config.get("system_site_packages", False)}


def system_site_packages_enabled():
    return _environment_config()["system_site_packages"]


def configure_system_site_packages(enabled, venv=None):
    if type(enabled) is not bool:
        raise UserError("ENVIRONMENT_CONFIG_INVALID", "system_site_packages 必须是明确的 true 或 false。")
    config = _environment_config()
    target_venv = venv or config["venv"]
    target_path = safe_project_path(target_venv, "虚拟环境")
    if target_path.exists():
        raise UserError(
            "ENVIRONMENT_POLICY_LOCKED",
            "目标项目虚拟环境已存在，不能改变它的基础包继承策略。",
            "保留现有环境；若要切换策略，请用 --venv 指定尚不存在的新路径。",
            {"venv": target_venv, "system_site_packages": config["system_site_packages"]},
        )
    config["system_site_packages"] = enabled
    config["venv"] = target_path.relative_to(Path.cwd().resolve(strict=True)).as_posix()
    atomic_write_bytes(
        safe_output_path(ENVIRONMENT_CONFIG_PATH.as_posix(), "项目环境配置"),
        (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return config


def save_environment_selection(venv):
    config_path = safe_output_path(ENVIRONMENT_CONFIG_PATH.as_posix(), "项目环境配置")
    relative = safe_project_path(venv, "虚拟环境").relative_to(Path.cwd().resolve(strict=True)).as_posix()
    existing = _environment_config()
    value = {
        "schema_version": 1,
        "venv": relative,
        "system_site_packages": existing["system_site_packages"],
    }
    atomic_write_bytes(config_path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def selected_environment_venv(requested=None):
    if requested:
        return requested
    config_path = safe_output_path(ENVIRONMENT_CONFIG_PATH.as_posix(), "项目环境配置")
    if not config_path.is_file():
        return ".venv"
    return _environment_config()["venv"]


def virtualenv_python(venv):
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def remove_owned_environment(venv, identity):
    """Remove a newly reserved environment without following a replaced path."""
    try:
        current = venv.lstat()
    except FileNotFoundError:
        return True
    if venv.is_symlink() or not stat.S_ISDIR(current.st_mode):
        return False
    if (current.st_dev, current.st_ino) != identity:
        return False
    shutil.rmtree(venv)
    return True


def installed_environment(python):
    script = (
        "import importlib.metadata,json; "
        "print(json.dumps(sorted([{'name':d.metadata.get('Name',''),'version':d.version} "
        "for d in importlib.metadata.distributions() if d.metadata.get('Name') and d.version], "
        "key=lambda p:p['name'].lower())))"
    )
    result = run_environment_process([python, "-c", script], "已安装包清单读取", timeout=60)
    try:
        packages = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise UserError("ENVIRONMENT_INVENTORY_INVALID", "Python 没有返回有效的已安装包清单。") from error
    if not isinstance(packages, list):
        raise UserError("ENVIRONMENT_INVENTORY_INVALID", "Python 已安装包清单格式无效。")
    return packages


def compare_environment(python, metadata, requirements_path=None):
    identity = python_identity(python)
    expected_identity = metadata.get("python")
    expected = {
        distribution_name_key(item["name"]): item["version"]
        for item in metadata.get("packages", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("version"), str)
    }
    inherited_metadata = metadata.get("system_site_packages", {"enabled": False, "distributions": []})
    inherited_enabled = inherited_metadata.get("enabled") is True
    inherited = {
        distribution_name_key(item["name"]): item["version"]
        for item in inherited_metadata.get("distributions", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("version"), str)
    }
    inherited = {name: version for name, version in inherited.items() if name not in expected}
    actual = {}
    for item in installed_environment(python):
        name = distribution_name_key(item["name"])
        if name in {"pip", "setuptools", "wheel"}:
            continue
        actual[name] = item["version"]
    missing = sorted(name for name in expected if name not in actual)
    changed = sorted(
        [
            {"name": name, "expected": expected[name], "actual": actual[name]}
            for name in expected.keys() & actual.keys()
            if expected[name] != actual[name]
        ],
        key=lambda item: item["name"],
    )
    inherited_missing = sorted(name for name in inherited if name not in actual)
    inherited_changed = sorted(
        [
            {"name": name, "expected": inherited[name], "actual": actual[name]}
            for name in inherited.keys() & actual.keys()
            if inherited[name] != actual[name]
        ],
        key=lambda item: item["name"],
    )
    allowed = expected.keys() | inherited.keys()
    extra = sorted(name for name in actual if name not in allowed)
    identity_matches = identity == expected_identity
    input_matches = True
    if requirements_path is not None:
        current_digest, _ = requirements_content_digest(requirements_path)
        input_matches = current_digest == metadata.get("requirements_sha256")
    if inherited_metadata.get("enabled") is True:
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
            raise UserError("ENVIRONMENT_PROCESS_FAILED", f"项目 pip check 无法运行：{error}") from error
        pip_findings = sorted(
            line.strip()
            for line in (checked.stdout + "\n" + checked.stderr).splitlines()
            if line.strip() and line.strip() != "No broken requirements found."
        )
        pip_findings = [redact_environment_output(line, limit=2000) for line in pip_findings]
        inherited_findings = inherited_metadata.get("pip_check", {}).get("findings", [])
        new_pip_findings = sorted(set(pip_findings) - set(inherited_findings))
        pip_check_text = "No new dependency conflicts beyond the inherited base environment."
        if inherited_findings:
            pip_check_text += f" Inherited base warnings: {len(inherited_findings)}."
        if new_pip_findings:
            pip_check_text += " New conflicts: " + " | ".join(new_pip_findings)
    else:
        pip_findings = []
        inherited_findings = []
        new_pip_findings = []
        pip_check = run_environment_process([python, "-m", "pip", "check"], "依赖一致性检查", timeout=120)
        pip_check_text = pip_check.stdout or "No broken requirements found."
    return {
        "python": identity,
        "python_matches": identity_matches,
        "requirements_match": input_matches,
        "system_site_packages": inherited_enabled,
        "missing": missing,
        "version_mismatch": changed,
        "inherited_missing": inherited_missing,
        "inherited_version_mismatch": inherited_changed,
        "inherited_pip_check": inherited_findings,
        "new_pip_check": new_pip_findings,
        "extra": extra,
        "pip_check": redact_environment_output(pip_check_text),
        "matches": identity_matches and input_matches and not missing and not changed and not inherited_missing and not inherited_changed and not extra and not new_pip_findings,
    }


def environment_operation(request):
    operation = request["operation"]
    runtime = request.get("runtime", {})
    base_python = runtime.get("python_executable") or "python3"
    explicit_python = request.get("python_explicit") is True
    stream_output = request.get("interactive_progress") is True
    inherit_system = system_site_packages_enabled()
    if operation == "environment_configure":
        enabled = request.get("system_site_packages")
        if type(enabled) is not bool:
            raise UserError("ENVIRONMENT_CONFIG_INVALID", "必须显式提供 system_site_packages 布尔值。")
        config = configure_system_site_packages(enabled, request.get("venv"))
        return {
            "venv": config["venv"],
            "system_site_packages": config["system_site_packages"],
            "note": "策略仅影响下一次创建的项目虚拟环境；已有环境不会被改写。",
        }
    if operation == "environment_lock":
        selected_venv = safe_project_path(selected_environment_venv(), "虚拟环境")
        resolver_python = base_python
        selected_python = virtualenv_python(selected_venv)
        if not explicit_python and selected_venv.is_dir() and selected_python.is_file():
            resolver_python = selected_python.as_posix()
        result = resolve_environment_lock(resolver_python, request.get("requirements") or "requirements.txt", request.get("refresh_lock") is True, stream_output, inherit_system)
        return {
            "lock_file": ENVIRONMENT_LOCK_PATH.as_posix(),
            "metadata_file": ENVIRONMENT_METADATA_PATH.as_posix(),
            "requirements_file": result["metadata"]["requirements_file"],
            "package_count": len(result["packages"]),
            "cached": result["cached"],
            "python": result["metadata"]["python"],
            "packages": result["packages"],
            "system_site_packages": result["metadata"].get("system_site_packages"),
            "note": "锁定 Python ABI、解析后的包版本与 wheel SHA-256；不会记录索引地址或凭证。",
        }
    if operation == "environment_setup":
        venv_relative = selected_environment_venv(request.get("venv"))
        venv = safe_project_path(venv_relative, "虚拟环境")
        if venv.exists() and (not venv.is_dir() or not virtualenv_python(venv).is_file()):
            raise UserError("ENVIRONMENT_ALREADY_EXISTS", f"目标路径已存在但不是可用的虚拟环境：{venv_relative}", "保留现有目录；选择新的 --venv 路径，或手动检查该目录。")
        resolver_python = base_python
        selected_python = virtualenv_python(venv)
        if not explicit_python and venv.is_dir() and selected_python.is_file():
            resolver_python = selected_python.as_posix()
        if inherit_system:
            resolver_python = base_python
        resolved = resolve_environment_lock(resolver_python, request.get("requirements") or "requirements.txt", request.get("refresh_lock") is True, stream_output, inherit_system)
        if venv.exists():
            check = compare_environment(virtualenv_python(venv).as_posix(), resolved["metadata"], resolved["metadata"]["requirements_file"])
            if not check["matches"]:
                raise UserError(
                    "ENVIRONMENT_DRIFT",
                    f"现有虚拟环境不符合依赖锁：缺少 {len(check['missing'])} 个、版本不符 {len(check['version_mismatch'])} 个、多出 {len(check['extra'])} 个包。",
                    "为避免覆盖用户环境，Model-Maven 没有卸载或改写它。选择新的 --venv 路径，或先人工备份并移走现有环境。",
                    check,
                )
            save_environment_selection(venv_relative)
            return {
                "venv": venv_relative,
                "python": virtualenv_python(venv).as_posix(),
                "created": False,
                "package_count": len(resolved["packages"]),
                "cached_lock": resolved["cached"],
                "system_site_packages": inherit_system,
                "environment_check": check,
                "note": "现有环境与锁完全匹配，已复用。",
            }
        venv.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Reserve the absent destination exclusively. On any failed setup,
            # only this exact directory may be rolled back; existing or
            # concurrently replaced user paths are never recursively removed.
            venv.mkdir()
        except FileExistsError as error:
            raise UserError(
                "ENVIRONMENT_ALREADY_EXISTS",
                f"目标路径已被占用：{venv_relative}",
                "保留现有目录；选择新的 --venv 路径，或检查并复用现有环境。",
            ) from error
        created_identity = venv.stat()
        created_identity = (created_identity.st_dev, created_identity.st_ino)
        try:
            venv_command = [base_python, "-m", "venv"]
            if inherit_system:
                venv_command.append("--system-site-packages")
            venv_command.append(str(venv))
            run_environment_process(venv_command, "项目虚拟环境创建", timeout=120)
            target_python = virtualenv_python(venv)
            if not target_python.is_file():
                raise UserError("ENVIRONMENT_CREATE_FAILED", f"虚拟环境未生成解释器：{venv_relative}")
            lock_text, _metadata = lock_file_paths()
            run_environment_process(
                [
                    target_python.as_posix(), "-m", "pip", "install", "--require-hashes",
                    "--only-binary=:all:", "--no-deps", "--pre", "--disable-pip-version-check", "-r", str(lock_text),
                ],
                "锁定依赖安装",
                stream_output=stream_output,
            )
            check = compare_environment(target_python.as_posix(), resolved["metadata"], resolved["metadata"]["requirements_file"])
            if not check["matches"]:
                raise UserError("ENVIRONMENT_VERIFY_FAILED", "依赖安装完成，但实际环境未通过锁文件核对。", "检查安装报告后重试；失败的本次新建环境会被移除，已有环境不会受影响。", check)
            save_environment_selection(venv_relative)
        except BaseException:
            # A newly-created environment is disposable until its package
            # inventory, pip check, and selection record all succeed.
            try:
                remove_owned_environment(venv, created_identity)
            except OSError:
                # Preserve the original setup/cancellation error. The path
                # identity check prevents cleanup from touching another owner.
                pass
            raise
        return {
            "venv": venv_relative,
            "python": target_python.as_posix(),
            "created": True,
            "package_count": len(resolved["packages"]),
            "cached_lock": resolved["cached"],
            "system_site_packages": inherit_system,
            "environment_check": check,
            "note": "项目虚拟环境已创建；锁定 wheel 安装、pip check 和包指纹核对均通过。",
        }
    if operation == "environment_check":
        _lock_text, metadata_path = lock_file_paths()
        if not metadata_path.is_file():
            raise UserError("ENVIRONMENT_LOCK_MISSING", "当前项目还没有 Model-Maven 依赖锁。", "先运行 m-model env lock 或 m-model env setup。")
        try:
            if metadata_path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("lock metadata is too large")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            lock_bytes = ENVIRONMENT_LOCK_PATH.read_bytes()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise UserError("ENVIRONMENT_LOCK_INVALID", "无法读取依赖锁或元数据。") from error
        validate_environment_lock(metadata, lock_bytes)
        venv_relative = selected_environment_venv(request.get("venv"))
        if explicit_python:
            target_python = base_python
        else:
            venv = safe_project_path(venv_relative, "虚拟环境")
            target_python = virtualenv_python(venv).as_posix() if venv.is_dir() and virtualenv_python(venv).is_file() else base_python
        check = compare_environment(target_python, metadata, metadata.get("requirements_file"))
        check["venv"] = venv_relative if target_python != base_python else None
        if not check["matches"]:
            raise UserError("ENVIRONMENT_DRIFT", "当前 Python 环境与 Model-Maven 依赖锁不一致。", "运行 m-model env setup；若要更新锁，先检查 requirements.txt 再使用 --update-lock。", check)
        return {"environment_check": check, "note": "Python ABI、依赖输入、包版本与依赖关系均符合锁文件。"}
    raise UserError("FEATURE_UNAVAILABLE", f"当前未实现操作：{operation}")

