"""Installed-environment snapshots, distinct from reconstructible locks."""

import datetime
import importlib.metadata
import json
import platform
import re
import sys

from ..protocol import UserError
from .common import distribution_name_key, environment_fingerprint

def snapshot_environment(torch_module=None, capture_timing="export_time"):
    packages = []
    grouped = {}
    issues = []
    distributions = importlib.metadata.distributions()
    for distribution in distributions:
        name = distribution.metadata.get("Name", "").strip()
        version = distribution.version
        if not name or not version:
            issues.append("有 distribution 缺少名称或版本，未加入版本快照。")
            continue
        key = distribution_name_key(name)
        direct_url = None
        try:
            direct_url = distribution.read_text("direct_url.json")
        except Exception:
            pass
        source_kind = "registry_or_unknown"
        if direct_url:
            try:
                source = json.loads(direct_url)
                if isinstance(source, dict):
                    if source.get("vcs_info"):
                        source_kind = "vcs"
                    elif isinstance(source.get("dir_info"), dict):
                        source_kind = (
                            "editable"
                            if source["dir_info"].get("editable") is True
                            else "local"
                        )
                    elif str(source.get("url", "")).startswith("file:"):
                        source_kind = "local"
                    else:
                        source_kind = "direct"
            except (TypeError, json.JSONDecodeError):
                source_kind = "direct_unknown"
        package = {"name": name, "version": version, "source_kind": source_kind}
        grouped.setdefault(key, []).append(package)
    if len(grouped) > 4096:
        raise UserError(
            "ENVIRONMENT_SNAPSHOT_TOO_LARGE",
            f"当前解释器包含 {len(grouped)} 个 distribution，超过 4096 项的导出清单上限。",
            "使用项目专用的 Python 环境减少无关包后重试；Model-Maven 不会自动卸载或改动包。",
        )
    ambiguous = []
    requirements = [
        "# Installed distribution versions captured from the selected Python environment.",
        "# This is an environment snapshot, not a resolver-generated or artifact-hashed lockfile.",
        "# Review private indexes, local/VCS/editable packages, platform and accelerator requirements before installing.",
    ]
    for key in sorted(grouped):
        entries = sorted(
            {(
                item["version"],
                item["source_kind"],
            ): item for item in grouped[key]}.values(),
            key=lambda item: (item["version"], item["source_kind"]),
        )
        packages.extend(entries)
        versions = {item["version"] for item in entries}
        source_kinds = {item["source_kind"] for item in entries}
        if len(versions) != 1 or len(source_kinds) != 1:
            ambiguous.append({
                "normalized_name": key,
                "versions": sorted(versions),
                "source_kinds": sorted(source_kinds),
            })
            continue
        item = entries[0]
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", item["name"]) and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", item["version"]
        ):
            requirements.append(f"{key}=={item['version']}")
        else:
            issues.append(f"distribution {key} 的名称或版本不适合写入 requirements 快照。")
    loaded_torch = torch_module
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    if loaded_torch is not None:
        torch_version = str(loaded_torch.__version__)
    pytorch_cuda_version = getattr(getattr(loaded_torch, "version", None), "cuda", None)
    python = {
        "implementation": sys.implementation.name,
        "version": sys.version.split()[0],
        "platform": sys.platform,
        "machine": platform.machine(),
        "abi": sys.implementation.cache_tag,
    }
    fingerprint = environment_fingerprint(
        python,
        torch_version,
        pytorch_cuda_version,
        packages,
        ambiguous,
        issues,
    )
    return {
        "schema_version": 1,
        "capture_kind": "installed_distribution_inventory",
        "capture_timing": capture_timing,
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "python": python,
        "pytorch_version": torch_version,
        "pytorch_cuda_version": pytorch_cuda_version,
        "fingerprint_algorithm": "canonical-json-sha256-v1",
        "fingerprint_sha256": fingerprint,
        "distribution_count": len(packages),
        "distributions": packages,
        "ambiguous_distribution_names": ambiguous,
        "issues": issues,
        "requirements_snapshot": "\n".join(requirements) + "\n",
        "limitations": [
            "版本列表来自当前解释器的 installed metadata，不会解析项目的最小依赖闭包。",
            "不包含 wheel/sdist 文件哈希、索引 URL、私有凭证或数据/模型外部权重。",
            "local、VCS、editable 或有同名多版本的包需要用户检查来源；requirements.snapshot.txt 不是可无条件安装的 lockfile。",
        ],
    }

