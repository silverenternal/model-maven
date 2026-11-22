"""Project-local extension removal planning and application."""

import hashlib
import importlib.metadata
import json
import os
import re
import stat
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from ..protocol import UserError


@dataclass(frozen=True)
class ComponentInstallRuntime:
    safe_project_path: Callable[[str, str], Path]
    distribution_name_key: Callable[[str], str]
    run_environment_process: Callable[..., Any]
    wheel_runtime_requirements: Callable[[Path], dict]
    sha256_file: Callable[[Path], str]
    discover_components: Callable[[dict], dict]
    project_root: Path
    python_executable: str


@dataclass(frozen=True)
class ComponentUninstallRuntime:
    component_json_files: Callable[[Path], Iterable]
    component_reference_paths: Callable[[Any, str, str, str], list]
    component_source_imports: Callable[[Path, dict, Path | None], tuple]


def _removal_tree_evidence(root):
    """Fingerprint the exact regular files and directories proposed for removal."""
    digest = hashlib.sha256()
    count = 0
    total = 0
    def walk_error(error):
        raise error
    try:
        for current, directories, files in os.walk(root, topdown=True, onerror=walk_error, followlinks=False):
            directories.sort()
            files.sort()
            for name in directories:
                path = Path(current) / name
                info = path.lstat()
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError(f"扩展目录包含符号链接或非目录：{path}")
                digest.update(b"D\0" + os.fsencode(path.relative_to(root)) + b"\0")
            for name in files:
                path = Path(current) / name
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError(f"扩展目录包含符号链接或非普通文件：{path}")
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError(f"扩展目录包含非普通文件：{path}")
                    file_hash = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        file_hash.update(chunk)
                    after = os.fstat(stream.fileno())
                    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                        raise ValueError(f"计算摘要时文件发生变化：{path}")
                    digest.update(b"F\0" + os.fsencode(path.relative_to(root)) + b"\0" + file_hash.digest())
                    count += 1
                    total += after.st_size
    except (OSError, ValueError) as error:
        raise UserError("COMPONENT_STORE_UNREADABLE", f"无法核对待移除扩展目录：{error}", "检查扩展目录后重新预览；没有移除组件。") from error
    return {"algorithm": "sha256-removal-tree-v1", "sha256": digest.hexdigest(), "files": count, "bytes": total}


def component_install(runtime, request):
    source_value = request.get("path")
    if not isinstance(source_value, str) or not source_value:
        raise UserError("COMPONENT_INSTALL_PATH_INVALID", "扩展包路径必须是项目内相对目录。")
    source = runtime.safe_project_path(source_value, "扩展包目录")
    if not source.is_dir():
        raise UserError("COMPONENT_INSTALL_PATH_INVALID", f"扩展包路径不是目录：{source_value}")
    try:
        source.relative_to(runtime.project_root / ".model-maven")
    except ValueError:
        pass
    else:
        raise UserError("COMPONENT_INSTALL_PATH_INVALID", "不能从 Model-Maven 自己的扩展存储目录再次安装包。")
    pyproject = source / "pyproject.toml"
    if pyproject.is_symlink() or not pyproject.is_file():
        raise UserError("COMPONENT_INSTALL_PACKAGE_INVALID", "扩展目录必须包含普通文件 pyproject.toml。")

    no_build_isolation = request.get("no_build_isolation") is True
    if no_build_isolation:
        try:
            try:
                import tomllib
            except ImportError:
                from pip._vendor import tomli as tomllib
            project_config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except Exception as error:
            raise UserError("COMPONENT_BUILD_CONFIG_INVALID", "无法读取扩展 pyproject.toml，未启动构建。", "检查 TOML 语法；使用 --no-build-isolation 时，构建依赖必须能在构建前完成检查。") from error
        has_build_system = "build-system" in project_config
        build_system = project_config.get("build-system", {})
        if not isinstance(build_system, dict):
            raise UserError("COMPONENT_BUILD_CONFIG_INVALID", "pyproject.toml 的 [build-system] 必须是表格。")
        if has_build_system and "requires" not in build_system:
            raise UserError("COMPONENT_BUILD_CONFIG_INVALID", "pyproject.toml 的 [build-system] 必须声明 requires。")
        build_requirements = build_system.get("requires", ["setuptools>=40.8.0"])
        if not isinstance(build_requirements, list) or not all(isinstance(item, str) for item in build_requirements):
            raise UserError("COMPONENT_BUILD_CONFIG_INVALID", "[build-system].requires 必须是 PEP 508 字符串数组。")
        from pip._vendor.packaging.markers import default_environment
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.version import Version
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
        for raw in build_requirements:
            try:
                requirement = Requirement(raw)
            except Exception as error:
                raise UserError("COMPONENT_BUILD_CONFIG_INVALID", f"无效的构建依赖声明：{raw}", "修正 [build-system].requires 中的 PEP 508 声明。") from error
            if requirement.url is not None:
                raise UserError(
                    "COMPONENT_BUILD_REQUIREMENT_UNCHECKABLE",
                    f"不能核实直接 URL 构建依赖是否与当前环境中的 distribution 相同：{raw}",
                    "改用版本约束的构建依赖，或移除 --no-build-isolation 让 pip 创建隔离构建环境。",
                )
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            key = runtime.distribution_name_key(requirement.name)
            version = installed.get(key)
            satisfied = version is not None and key not in duplicates
            if satisfied and requirement.specifier:
                try:
                    satisfied = requirement.specifier.contains(Version(version), prereleases=True)
                except Exception:
                    satisfied = False
            if not satisfied:
                missing.append({"requirement": raw, "installed_version": version, "ambiguous": key in duplicates})
        if missing:
            raise UserError(
                "COMPONENT_BUILD_REQUIREMENTS_MISSING",
                "当前 Python 环境缺少扩展构建依赖；已在构建前停止。",
                "安装这些构建依赖，或移除 --no-build-isolation 以使用 pip 的隔离构建。",
                {"missing": missing},
            )

    metadata = runtime.project_root / ".model-maven"
    extension_root = metadata / "extensions"
    for path, label in ((metadata, "项目元数据"), (extension_root, "项目扩展存储")):
        if path.is_symlink():
            raise UserError("COMPONENT_INSTALL_PATH_UNSAFE", f"{label}路径不能是符号链接。")
        if path.exists() and not path.is_dir():
            raise UserError("COMPONENT_INSTALL_PATH_UNSAFE", f"{label}路径不是目录。")
        path.mkdir(mode=0o700, exist_ok=True)

    runtime_options = request.get("runtime", {})
    timeout = runtime_options.get("timeout_seconds", 1800) if isinstance(runtime_options, dict) else 1800
    if type(timeout) is not int or timeout < 1:
        raise UserError("COMPONENT_INSTALL_TIMEOUT_INVALID", "扩展构建超时必须是正整数秒数。")
    progress = request.get("interactive_progress") is True
    with tempfile.TemporaryDirectory(prefix="model-maven-component-wheel-") as temporary:
        wheels = Path(temporary) / "wheels"
        wheels.mkdir()
        if progress:
            sys.stderr.write("MODEL_MAVEN_PROGRESS {\"message\":\"正在按 pyproject.toml 构建扩展 wheel…\"}\n")
            sys.stderr.flush()
        runtime.run_environment_process(
            [runtime.python_executable, "-m", "pip", "wheel", "--disable-pip-version-check", "--no-deps"]
            + (["--no-build-isolation"] if no_build_isolation else [])
            + ["--wheel-dir", str(wheels), str(source)],
            "扩展 wheel 构建",
            timeout=timeout,
            stream_output=progress,
            source_build=True,
        )
        wheel_files = sorted(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise UserError("COMPONENT_BUILD_OUTPUT_INVALID", f"扩展构建应生成一个 wheel，实际得到 {len(wheel_files)} 个。")
        wheel = wheel_files[0]
        wheel_size = wheel.stat().st_size
        wheel_metadata = runtime.wheel_runtime_requirements(wheel)
        wheel_hash = runtime.sha256_file(wheel)
        target = extension_root / wheel_hash
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise UserError("COMPONENT_INSTALL_PATH_UNSAFE", "相同 wheel 哈希对应的扩展存储路径不是普通目录。")
            already_installed = True
        else:
            already_installed = False
            stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=extension_root))
            try:
                if progress:
                    sys.stderr.write("MODEL_MAVEN_PROGRESS {\"message\":\"wheel 已构建，正在复制到项目隔离扩展存储…\"}\n")
                    sys.stderr.flush()
                runtime.run_environment_process(
                    [runtime.python_executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "--target", str(stage), str(wheel)],
                    "项目扩展包安装",
                    timeout=timeout,
                    stream_output=progress,
                )
                staged = runtime.discover_components({"paths": [str(stage)], "_include_project_extensions": False})
                if not staged["components"] or staged["issues"]:
                    raise UserError(
                        "COMPONENT_PACKAGE_INVALID",
                        "wheel 中未发现有效的 Model-Maven 组件入口或静态 manifest。",
                        "检查 model_maven.components entry point 与 __model_maven__/<entry-point-name>.json。",
                        {"issues": staged["issues"]},
                    )
                try:
                    stage.rename(target)
                except FileExistsError:
                    raise UserError("COMPONENT_INSTALL_CONFLICT", "相同 wheel 正在被另一个进程安装，请稍后重试。")
            except Exception:
                if stage.exists() and not stage.is_symlink():
                    import shutil
                    shutil.rmtree(stage, ignore_errors=True)
                raise

    discovered = runtime.discover_components({"paths": [str(target)], "_include_project_extensions": False})
    if not discovered["components"] or discovered["issues"]:
        # An existing hash directory may be a prior, still usable installation.
        # A failed rediscovery must never turn an idempotent install attempt into
        # destructive cleanup of state this invocation did not create.
        if not already_installed:
            import shutil
            shutil.rmtree(target, ignore_errors=True)
        raise UserError(
            "COMPONENT_PACKAGE_INVALID",
            "安装后的组件包未通过静态发现检查。",
            "检查 entry point、manifest、入口模块和依赖 fixture 文件。",
            {"issues": discovered["issues"]},
        )
    components = []
    for item in discovered["components"]:
        manifest = item["manifest"]
        components.append({
            "component_id": manifest.get("component_id"),
            "version": manifest.get("component_version"),
            "kind": manifest.get("kind"),
            "distribution": item["package_source"].get("distribution"),
        })
    return {
        "path": source.relative_to(runtime.project_root).as_posix(),
        "site_packages": target.relative_to(runtime.project_root).as_posix(),
        "wheel_sha256": wheel_hash,
        "wheel_size_bytes": wheel_size,
        "distribution": wheel_metadata["name"],
        "distribution_version": wheel_metadata["version"],
        "runtime_dependencies": wheel_metadata["requirements"],
        "components": components,
        "already_installed": already_installed,
        "environment_modified": False,
    }


def component_uninstall(runtime, request):
    requested = request.get("id")
    if not isinstance(requested, str) or requested.count("@") != 1:
        raise UserError("COMPONENT_UNINSTALL_ID_INVALID", "卸载需要精确组件版本 namespace/name@MAJOR.MINOR.PATCH。")
    component_id, version = requested.rsplit("@", 1)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}/[A-Za-z0-9][A-Za-z0-9_-]{0,63}", component_id) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise UserError("COMPONENT_UNINSTALL_ID_INVALID", "组件 ID 或精确版本格式无效。")
    apply = request.get("apply") is True
    force = request.get("force") is True
    if force and not apply:
        raise UserError("COMPONENT_UNINSTALL_FLAGS_INVALID", "--force 需要与 --apply 一起使用。")
    project_root = Path.cwd().resolve(strict=True)
    metadata = project_root / ".model-maven"
    registry = metadata / "components"
    extension_root = metadata / "extensions"
    for path, label in ((metadata, "项目元数据"), (registry, "组件注册表"), (extension_root, "项目扩展存储")):
        if path.is_symlink():
            raise UserError("COMPONENT_STORE_UNSAFE", f"{label}不能是符号链接：{path}")
        if path.exists() and not path.is_dir():
            raise UserError("COMPONENT_STORE_UNSAFE", f"{label}不是目录：{path}")

    manifests = []
    for path, value in runtime.component_json_files(registry):
        if value.get("component_id") == component_id and value.get("component_version") == version:
            contents = path.read_bytes()
            manifests.append((path, value, contents, hashlib.sha256(contents).hexdigest()))
    if len(manifests) != 1:
        if not manifests:
            raise UserError("COMPONENT_NOT_REGISTERED", f"项目中没有注册 {component_id}@{version}。")
        raise UserError("COMPONENT_REGISTRY_AMBIGUOUS", f"组件 {component_id}@{version} 出现多个注册清单，停止移除。")
    manifest_path, manifest, manifest_bytes, manifest_hash = manifests[0]
    package_source = manifest.get("package_source")
    distribution = package_source.get("distribution") if isinstance(package_source, dict) else None
    raw_distribution_root = package_source.get("distribution_root") if isinstance(package_source, dict) else None
    installed_root = None
    if isinstance(raw_distribution_root, str):
        raw_root_path = Path(raw_distribution_root)
        try:
            lexical_relative = raw_root_path.relative_to(extension_root)
        except ValueError:
            lexical_relative = None
        if lexical_relative is not None:
            if any(part in {".", ".."} for part in lexical_relative.parts):
                raise UserError("COMPONENT_INSTALL_PATH_UNSAFE", "组件来源路径不能包含目录跳转。")
            current = extension_root
            for part in lexical_relative.parts:
                current = current / part
                if current.is_symlink():
                    raise UserError("COMPONENT_INSTALL_PATH_UNSAFE", "组件来源路径不能经过符号链接。")
        try:
            candidate_root = raw_root_path.resolve(strict=True)
        except OSError:
            candidate_root = None
        if candidate_root is not None:
            try:
                relative = candidate_root.relative_to(extension_root.resolve(strict=True))
            except (OSError, ValueError):
                pass
            else:
                if lexical_relative is None:
                    raise UserError("COMPONENT_INSTALL_PATH_UNSAFE", "组件来源经外部路径映射进入项目扩展存储，停止删除。")
                if len(relative.parts) != 1 or not re.fullmatch(r"[a-f0-9]{64}", relative.name):
                    raise UserError("COMPONENT_INSTALL_PATH_UNSAFE", "组件来源不在有效的内容寻址扩展目录中。")
                if candidate_root.is_symlink() or not candidate_root.is_dir():
                    raise UserError("COMPONENT_INSTALL_PATH_UNSAFE", "组件扩展目录不是普通目录。")
                for current, directories, files in os.walk(candidate_root, followlinks=False):
                    for name in directories + files:
                        if (Path(current) / name).is_symlink():
                            raise UserError("COMPONENT_INSTALL_PATH_UNSAFE", "组件扩展目录含符号链接，停止移除。")
                installed_root = candidate_root

    remaining_components = []
    if installed_root is not None:
        for path, value in runtime.component_json_files(registry):
            if path == manifest_path:
                continue
            source = value.get("package_source")
            root_value = source.get("distribution_root") if isinstance(source, dict) else None
            if not isinstance(root_value, str):
                continue
            try:
                same_root = Path(root_value).resolve(strict=True) == installed_root
            except OSError:
                same_root = False
            if same_root:
                remaining_components.append({
                    "id": value.get("component_id"),
                    "version": value.get("component_version"),
                    "path": str(path.relative_to(project_root)),
                })

    project_config_references = []
    variant_references = []
    project_config = metadata / "project.json"
    try:
        config_metadata = project_config.lstat()
    except FileNotFoundError:
        config_metadata = None
    except OSError as error:
        raise UserError("COMPONENT_STORE_UNREADABLE", f"无法检查项目配置 {project_config}：{error}") from error
    if config_metadata is not None:
        if project_config.is_symlink() or not project_config.is_file() or config_metadata.st_size > 64 * 1024:
            raise UserError("COMPONENT_STORE_UNSAFE", "项目配置必须是小于 64 KiB 的普通文件，不能通过符号链接读取。")
        try:
            config_value = json.loads(project_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UserError("COMPONENT_STORE_UNREADABLE", f"无法读取项目配置 {project_config}：{error}") from error
        project_config_references.extend(
            runtime.component_reference_paths(
                config_value, component_id, version, project_config.relative_to(project_root).as_posix()
            )
        )
    for data_dir in (metadata / "variants", metadata / "runs"):
        for path, value in runtime.component_json_files(data_dir):
            relative = path.relative_to(project_root).as_posix()
            variant_references.extend(runtime.component_reference_paths(value, component_id, version, relative))
    unique_references = {(item["path"], item["component"]): item for item in variant_references}
    variant_references = sorted(unique_references.values(), key=lambda item: (item["path"], item["component"]))
    project_config_references = sorted(
        {(item["path"], item["component"]): item for item in project_config_references}.values(),
        key=lambda item: (item["path"], item["component"]),
    )

    source_imports, unreadable_sources = runtime.component_source_imports(
        project_root, manifest, installed_root
    )
    active_runs = request.get("active_runs", [])
    if not isinstance(active_runs, list):
        raise UserError("COMPONENT_RUN_STATE_INVALID", "活动运行状态必须是数组。")
    active_runs = [item for item in active_runs if isinstance(item, dict)]
    active_runs.sort(key=lambda item: str(item.get("id", "")))
    wheel_removed = installed_root is not None and not remaining_components
    removal_tree = _removal_tree_evidence(installed_root) if wheel_removed else None
    requires_force = bool(
        project_config_references
        or variant_references
        or (wheel_removed and (source_imports or unreadable_sources))
    )
    plan = {
        "component_id": component_id,
        "component_version": version,
        "manifest_path": str(manifest_path.relative_to(project_root)),
        "manifest_sha256": manifest_hash,
        "distribution": distribution,
        "extension_path": str(installed_root.relative_to(project_root)) if installed_root else None,
        "wheel_removed": wheel_removed,
        "removal_tree": removal_tree,
        "extension_retained": installed_root is not None and bool(remaining_components),
        "remaining_components": remaining_components,
        "project_config_references": project_config_references,
        "variant_references": variant_references,
        "source_imports": source_imports,
        "unreadable_sources": unreadable_sources,
        "active_runs": active_runs,
        "requires_force": requires_force,
        "applied": False,
        "note": "移除只删除注册清单、对应 Provider 验证记录及不再被其他组件使用的项目 wheel；变体和 Python 源码保持原样。",
    }
    plan_revision = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    plan["plan_revision"] = plan_revision
    if active_runs:
        raise UserError("COMPONENT_ACTIVE_RUNS", "项目有运行中或状态未知的任务，不能移除组件。", "停止活动运行并核实 unknown 进程后重试。", plan)
    if apply and requires_force and not force:
        raise UserError(
            "COMPONENT_IN_USE",
            "该组件仍被项目配置、保存的变体或项目 Python 源码引用。",
            "检查预览中的引用；确认愿意让这些入口失效后使用 --apply --force。原配置、变体和源码不会被改写。",
            plan,
        )
    if not apply:
        return plan

    expected_plan = request.get("if_plan")
    if expected_plan is not None and expected_plan != plan_revision:
        raise UserError("COMPONENT_PLAN_CONFLICT", "组件或其影响与确认的移除计划不一致。", "重新预览后再确认；没有移除组件。", plan)

    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
        raise UserError("COMPONENT_CHANGED_DURING_UNINSTALL", "预览后组件清单发生变化，已取消移除。", "重新运行 component uninstall 生成新预览。")
    current = metadata
    for part in manifest_path.relative_to(metadata).parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise UserError("COMPONENT_STORE_UNSAFE", "组件清单父目录在移除前变为非目录或符号链接。")
    verification_path = metadata / "component-verifications" / component_id.split("/")[0] / component_id.split("/")[1] / f"{version}.json"
    current = metadata
    for part in verification_path.relative_to(metadata).parts[:-1]:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise UserError("COMPONENT_STORE_UNSAFE", "Provider 验证记录目录包含符号链接或非目录路径。")
    if verification_path.is_symlink() or (verification_path.exists() and not verification_path.is_file()):
        raise UserError("COMPONENT_STORE_UNSAFE", "Provider 验证记录必须是普通文件。")
    moved_wheel = None
    if plan["wheel_removed"] and installed_root is not None:
        staging_root = Path(tempfile.mkdtemp(prefix=f".uninstall-{installed_root.name}-", dir=extension_root))
        staging = staging_root / "wheel"
        try:
            os.replace(installed_root, staging)
            moved_wheel = (staging, installed_root, staging_root)
        except OSError as error:
            import shutil
            shutil.rmtree(staging_root, ignore_errors=True)
            raise UserError("COMPONENT_UNINSTALL_FAILED", f"无法暂存项目扩展目录：{error}") from error
    try:
        manifest_path.unlink()
        if verification_path.exists():
            verification_path.unlink()
    except OSError as error:
        if manifest_path.exists() is False:
            temporary_manifest = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{manifest_path.name}.restore-",
                    dir=manifest_path.parent,
                    delete=False,
                ) as stream:
                    stream.write(manifest_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                    temporary_manifest = Path(stream.name)
                os.replace(temporary_manifest, manifest_path)
            except OSError as restore_error:
                if temporary_manifest is not None:
                    temporary_manifest.unlink(missing_ok=True)
                raise UserError(
                    "COMPONENT_UNINSTALL_ROLLBACK_FAILED",
                    f"卸载未完成，且无法恢复原组件清单：{restore_error}",
                    "检查组件注册目录权限和文件系统状态；不要手工删除剩余暂存内容。",
                ) from restore_error
        if moved_wheel is not None and moved_wheel[0].exists():
            os.replace(moved_wheel[0], moved_wheel[1])
            import shutil
            shutil.rmtree(moved_wheel[2], ignore_errors=True)
        raise UserError("COMPONENT_UNINSTALL_FAILED", f"无法移除组件注册信息，已尝试恢复原文件：{error}") from error
    if moved_wheel is not None:
        import shutil
        try:
            shutil.rmtree(moved_wheel[2])
        except OSError as error:
            plan["cleanup_pending"] = str(moved_wheel[2].relative_to(project_root))
            plan["cleanup_error"] = str(error)
    plan["applied"] = True
    plan["note"] = "组件注册信息已移除；变体、源码、训练运行记录与 Python 环境未被修改。"
    return plan
