"""Dependency closure planning for portable extension bundles."""

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from ..protocol import UserError
from .catalog import ExternalDistributionResource


@dataclass(frozen=True)
class ExtensionBundleRuntime:
    verify_component_distribution: Callable[[dict], str | None]
    distribution_name_key: Callable[[str], str]
    safe_distribution_file: Callable[..., Path]
    sha256_file: Callable[[Path], str]
    search_path_entries: tuple[str, ...]
    current_directory: str


def extension_bundle_plan(runtime, request):
    components = request.get("components", [])
    if not isinstance(components, list) or len(components) > 64:
        raise UserError("EXTENSION_BUNDLE_INVALID", "扩展包清单必须是最多 64 项的数组。")
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
            from pip._vendor.packaging.version import Version
        except ImportError:
            try:
                from setuptools._vendor.packaging.requirements import Requirement
                from setuptools._vendor.packaging.version import Version
            except ImportError as error:
                raise UserError(
                    "EXTENSION_BUNDLE_PACKAGING_UNAVAILABLE",
                    "当前 Python 缺少 PEP 508 依赖解析器，无法安全计算扩展依赖闭包。",
                    "在所选 Python 环境安装 packaging 后重试；不会猜测或忽略依赖声明。",
                ) from error

    search_paths = []
    for item in runtime.search_path_entries:
        try:
            root = Path(item or runtime.current_directory).resolve(strict=True)
        except OSError:
            continue
        if root.is_dir() and str(root) not in search_paths:
            search_paths.append(str(root))
    for item in components:
        if not isinstance(item, dict) or not isinstance(item.get("manifest"), dict):
            continue
        package_source = item["manifest"].get("package_source")
        raw_root = package_source.get("distribution_root") if isinstance(package_source, dict) else None
        if not isinstance(raw_root, str):
            continue
        try:
            root = Path(raw_root).resolve(strict=True)
        except OSError:
            continue
        if root.is_dir() and str(root) not in search_paths:
            search_paths.append(str(root))
    installed = {}
    seen_distributions = set()
    for distribution in importlib.metadata.distributions(path=search_paths):
        name = distribution.metadata.get("Name", "")
        if not name:
            continue
        try:
            root = Path(distribution.locate_file("")).resolve(strict=True)
        except OSError:
            continue
        identity = (runtime.distribution_name_key(name), distribution.version, str(root))
        if identity in seen_distributions:
            continue
        seen_distributions.add(identity)
        installed.setdefault(runtime.distribution_name_key(name), []).append((distribution, root))

    packages = {}
    queue = []
    for item in components:
        if not isinstance(item, dict) or not isinstance(item.get("manifest"), dict):
            raise UserError("EXTENSION_BUNDLE_INVALID", "扩展组件描述缺少 manifest。")
        manifest = item["manifest"]
        package_source = manifest.get("package_source")
        if not isinstance(package_source, dict):
            continue
        root_value = runtime.verify_component_distribution(manifest)
        if not isinstance(root_value, str):
            raise UserError("EXTENSION_BUNDLE_INVALID", "扩展来源未返回可验证的 distribution 目录。")
        root = Path(root_value).resolve(strict=True)
        name = package_source.get("distribution")
        version = package_source.get("distribution_version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise UserError("EXTENSION_BUNDLE_INVALID", "扩展 distribution 名称或版本无效。")
        matches = [
            distribution
            for distribution, candidate_root in installed.get(runtime.distribution_name_key(name), [])
            if distribution.version == version and candidate_root == root
        ]
        if len(matches) != 1:
            raise UserError(
                "EXTENSION_BUNDLE_AMBIGUOUS",
                f"扩展 {name}=={version} 在当前 Python 环境中不是唯一安装来源。",
                "清理同名重复安装并重新发现组件，避免导出错误版本。",
            )
        component_id = manifest.get("component_id")
        queue.append((matches[0], root, "extension", [component_id] if isinstance(component_id, str) else [], set(), name))

    runtime_requirements = {}
    processed_extras = {}
    total_file_count = 0
    total_bytes = 0

    def add_package(distribution, root, role, required_by, component_ids=None):
        nonlocal total_file_count, total_bytes
        name = distribution.metadata.get("Name", "")
        version = distribution.version
        identity = (runtime.distribution_name_key(name), version, str(root))
        package = packages.get(identity)
        if package is None:
            if len(packages) >= 64:
                raise UserError(
                    "EXTENSION_BUNDLE_TOO_MANY_PACKAGES",
                    "扩展及其 Python 依赖闭包超过 64 个 distribution。",
                    "减少扩展依赖层级后重新导出；不会生成缺少依赖的模型包。",
                )
            declared = distribution.files
            if declared is None:
                raise UserError(
                    "EXTENSION_BUNDLE_UNAVAILABLE",
                    f"依赖 {name}=={version} 缺少可枚举的安装文件清单。",
                    "重新安装为标准 wheel 后再导出；editable 安装和无 RECORD 的包不能作为独立包归档。",
                )
            declared_files = {PurePosixPath(str(path)).as_posix() for path in declared}
            package = {
                "distribution": name,
                "version": version,
                "role": role,
                "source_root": str(root),
                "components": [],
                "required_by": set(),
                "files": {},
                "omitted_files": [],
                "size_bytes": 0,
                "declared_files": declared_files,
            }
            packages[identity] = package
            for relative in sorted(declared_files):
                path = PurePosixPath(relative)
                if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                    package["omitted_files"].append({"path": relative, "reason": "outside_distribution_root"})
                    continue
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                if path.suffix == ".pth":
                    raise UserError(
                        "EXTENSION_BUNDLE_UNSUPPORTED_INSTALLER",
                        f"依赖 {name}=={version} 包含需要 site 启动处理的 .pth 文件。",
                        "使用不依赖 site 启动钩子的标准 wheel 后重试。",
                    )
                try:
                    file_path = runtime.safe_distribution_file(root, relative, declared_files, 256 * 1024 * 1024)
                except ExternalDistributionResource as error:
                    raise UserError(
                        "EXTENSION_BUNDLE_EXTERNAL_RESOURCE_UNSUPPORTED",
                        f"依赖 {name}=={version} 的文件 {relative} 指向 distribution 外部资源：{error}。",
                        "导出不会复制宿主机证书库、数据文件或其他外部目标；请改用自包含 wheel，或在部署环境单独提供该资源。",
                    ) from error
                except (OSError, ValueError) as error:
                    raise UserError(
                        "EXTENSION_BUNDLE_UNAVAILABLE",
                        f"无法安全归档 {name}=={version} 的文件 {relative}：{error}。",
                        "检查该 distribution 是否为完整、非 editable 的标准安装。",
                    ) from error
                size = file_path.stat().st_size
                package["files"][relative] = {
                    "source_path": str(file_path),
                    "sha256": runtime.sha256_file(file_path),
                    "size_bytes": size,
                }
                package["size_bytes"] += size
                total_file_count += 1
                total_bytes += size
                if total_file_count > 20000 or total_bytes > 2 * 1024 * 1024 * 1024:
                    raise UserError(
                        "EXTENSION_BUNDLE_TOO_LARGE",
                        "扩展及其 Python 依赖闭包超过 20000 个文件或 2 GiB。",
                        "缩小扩展依赖，或使用与运行环境匹配的外部发行渠道；不会生成不完整导出。",
                    )
            if not package["files"]:
                raise UserError("EXTENSION_BUNDLE_UNAVAILABLE", f"{name}=={version} 没有可归档文件。")
        elif role == "extension":
            package["role"] = "extension"
        package["required_by"].update(required_by)
        if component_ids:
            package["components"].extend(item for item in component_ids if item not in package["components"])
        return package

    while queue:
        distribution, root, role, required_by, requested_extras, declared_name = queue.pop(0)
        name = distribution.metadata.get("Name", "")
        version = distribution.version
        key = (runtime.distribution_name_key(name), version, str(root))
        package = add_package(distribution, root, role, required_by, required_by if role == "extension" else None)
        extras_key = key
        previous_extras = processed_extras.setdefault(extras_key, set())
        pending_extras = requested_extras - previous_extras
        if not previous_extras:
            pending_extras.add("")
        if not pending_extras:
            continue
        previous_extras.update(pending_extras)
        requires = distribution.requires or []
        for raw_requirement in requires:
            try:
                requirement = Requirement(raw_requirement)
            except Exception as error:
                raise UserError(
                    "EXTENSION_BUNDLE_INVALID_REQUIREMENT",
                    f"{name}=={version} 声明了无法解析的依赖 {raw_requirement!r}。",
                    "修正 distribution 的 PEP 508 Requires-Dist 元数据后重试。",
                ) from error
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in pending_extras
            ):
                continue
            if requirement.url:
                raise UserError(
                    "EXTENSION_BUNDLE_DIRECT_URL_UNSUPPORTED",
                    f"{name}=={version} 使用直接 URL 依赖 {requirement}，当前无法验证其来源与内容。",
                    "将该依赖发布为有版本的 wheel，并在 Requires-Dist 中使用版本约束。",
                )
            dependency_name = requirement.name
            dependency_key = runtime.distribution_name_key(dependency_name)
            parent = f"{name}=={version}"
            matches = installed.get(dependency_key, [])
            if len(matches) != 1:
                state = "未安装" if not matches else "存在多个安装来源"
                raise UserError(
                    "EXTENSION_BUNDLE_DEPENDENCY_UNAVAILABLE",
                    f"{parent} 需要 {requirement}，但当前环境中该依赖{state}。",
                    "安装唯一且满足版本约束的依赖后重新导出；不会生成缺少依赖的模型包。",
                )
            dependency, dependency_root = matches[0]
            try:
                dependency_version = Version(dependency.version)
                compatible = requirement.specifier.contains(dependency_version, prereleases=True)
            except Exception as error:
                raise UserError(
                    "EXTENSION_BUNDLE_DEPENDENCY_VERSION_INVALID",
                    f"无法验证 {dependency_name} 的安装版本 {dependency.version!r}。",
                    "重新安装带有有效 PEP 440 版本元数据的 distribution。",
                ) from error
            if not compatible:
                raise UserError(
                    "EXTENSION_BUNDLE_DEPENDENCY_VERSION_MISMATCH",
                    f"{parent} 需要 {requirement}，当前安装的是 {dependency_name}=={dependency.version}。",
                    "安装满足声明约束的版本后重新导出。",
                )
            if dependency_key == "torch":
                record = runtime_requirements.setdefault(
                    dependency_key,
                    {
                        "distribution": dependency.metadata.get("Name", dependency_name),
                        "version": dependency.version,
                        "constraints": [],
                        "required_by": set(),
                    },
                )
                if record["version"] != dependency.version:
                    raise UserError("EXTENSION_BUNDLE_AMBIGUOUS", "Torch 运行时依赖来源不唯一。")
                constraint = str(requirement.specifier)
                if constraint not in record["constraints"]:
                    record["constraints"].append(constraint)
                record["required_by"].add(parent)
                continue
            dependency_extras = set(requirement.extras)
            queue.append((dependency, dependency_root, "runtime_dependency", [parent], dependency_extras, dependency_name))

    if any(package["role"] == "extension" for package in packages.values()):
        torch_matches = installed.get("torch", [])
        if len(torch_matches) != 1:
            raise UserError(
                "EXTENSION_BUNDLE_RUNTIME_UNAVAILABLE",
                "无法唯一确定恢复模型所需的 PyTorch distribution。",
                "清理同名重复安装并使用运行该模型的 Python 环境重新导出。",
            )
        torch_distribution, _ = torch_matches[0]
        record = runtime_requirements.setdefault(
            "torch",
            {
                "distribution": torch_distribution.metadata.get("Name", "torch"),
                "version": torch_distribution.version,
                "constraints": [],
                "required_by": set(),
            },
        )
        record["required_by"].add("Model-Maven PyTorch runtime")

    result = []
    for package in packages.values():
        package["components"] = sorted(set(package["components"]))
        package["required_by"] = sorted(package["required_by"])
        package["files"] = [
            {"relative_path": path, **package["files"][path]}
            for path in sorted(package["files"])
        ]
        package["file_count"] = len(package["files"])
        del package["declared_files"]
        result.append(package)
    result.sort(key=lambda package: (package["distribution"].lower(), package["version"]))
    encoded_size = len(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    if encoded_size > 7 * 1024 * 1024:
        raise UserError(
            "EXTENSION_BUNDLE_PLAN_TOO_LARGE",
            "扩展文件清单超过 worker 报告限制。",
            "减少扩展及其依赖中的文件数量后再导出。",
        )
    external = []
    for requirement in runtime_requirements.values():
        requirement["constraints"].sort()
        requirement["required_by"] = sorted(requirement["required_by"])
        external.append(requirement)
    external.sort(key=lambda requirement: requirement["distribution"].lower())
    return {
        "packages": result,
        "runtime_requirements": external,
        "file_count": total_file_count,
        "size_bytes": total_bytes,
    }
