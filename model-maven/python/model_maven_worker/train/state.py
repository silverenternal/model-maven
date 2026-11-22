"""Per-step values and user-callable checkpoint state for custom training loops."""

import math
import hashlib
import importlib.metadata
import inspect
import json
import os
import random as python_random
import copy
from collections import Counter
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocol import UserError
from ..extensions.metadata import validate_state_format, validate_state_value


@dataclass(frozen=True)
class TrainingStateRuntime:
    torch: Any
    python_random: Any = python_random


class StepContext:
    """Small per-step view for a plain user callable; no framework base class required."""

    def __init__(self, step, total_steps):
        self.step = step
        self.total_steps = total_steps
        self._metrics = {}
        self._sample_ids_report = None
        self._sample_ids_callback = None

    def log(self, name, value):
        if not isinstance(name, str) or not name or len(name) > 128:
            raise UserError("STEP_RESULT_INVALID", "Metric 名称必须是 1..128 字符。")
        if len(self._metrics) >= 128 and name not in self._metrics:
            raise UserError("STEP_RESULT_INVALID", "每步最多记录 128 个 Metric。")
        self._metrics[name] = value

    def report_sample_ids(self, sample_ids, stable=True):
        """Report stable IDs consumed by this step; raw IDs are never retained."""
        if self._sample_ids_report is not None:
            raise UserError("STEP_RESULT_INVALID", "每个训练步骤只能报告一次 sample IDs。")
        self._sample_ids_report = sample_ids_report(sample_ids, stable)
        if self._sample_ids_callback is not None:
            self._sample_ids_callback(sample_ids, stable)


def sample_ids_report(sample_ids, stable=True):
    if type(stable) is not bool:
        raise UserError("STEP_RESULT_INVALID", "sample IDs 的 stable 标记必须是布尔值。")
    if not stable:
        return {"stable": False}
    if not isinstance(sample_ids, (list, tuple)) or not 1 <= len(sample_ids) <= 100_000:
        raise UserError(
            "STEP_RESULT_INVALID",
            "sample_ids 必须是 1..100000 项的稳定字符串列表。",
        )
    if any(
        not isinstance(sample_id, str)
        or not sample_id
        or len(sample_id) > 512
        or any(ord(character) < 32 for character in sample_id)
        for sample_id in sample_ids
    ):
        raise UserError(
            "STEP_RESULT_INVALID",
            "sample_ids 只能包含 1..512 字符且不含控制字符的非空字符串。",
        )
    encoded = json.dumps(
        list(sample_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "stable": True,
        "count": len(sample_ids),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


class SampleIdentityTracker:
    """Persist only a digest of ordered per-step sample IDs, never raw IDs."""

    def __init__(self, state=None, prior_steps=0):
        valid_state = (
            isinstance(state, dict)
            and state.get("schema_version") == 1
            and state.get("status") == "stable"
            and isinstance(state.get("sample_ids_sha256"), str)
            and len(state["sample_ids_sha256"]) == 64
            and type(state.get("sample_count")) is int
            and state["sample_count"] >= 0
            and type(state.get("steps_seen")) is int
            and state["steps_seen"] >= 0
            and type(state.get("reported_steps")) is int
            and state["reported_steps"] == state["steps_seen"]
        )
        self._stable = state is not None and valid_state
        if state is None and prior_steps == 0:
            self._stable = True
        self._sample_ids_sha256 = (
            state["sample_ids_sha256"] if valid_state else "0" * 64
        )
        self._sample_count = state["sample_count"] if valid_state else 0
        self._steps_seen = state["steps_seen"] if valid_state else max(0, prior_steps)
        self._reported_steps = state["reported_steps"] if valid_state else 0

    def record_step(self, report):
        self._steps_seen += 1
        if not isinstance(report, dict) or report.get("stable") is not True:
            self._stable = False
            return
        count = report.get("count")
        digest = report.get("sha256")
        if type(count) is not int or count < 1 or not isinstance(digest, str) or len(digest) != 64:
            self._stable = False
            return
        self._reported_steps += 1
        if not self._stable:
            return
        chain = f"{self._sample_ids_sha256}:{count}:{digest}".encode("ascii")
        self._sample_ids_sha256 = hashlib.sha256(chain).hexdigest()
        self._sample_count += count

    def checkpoint_state(self):
        return {
            "schema_version": 1,
            "status": "stable" if self._stable else "unknown",
            "sample_ids_sha256": self._sample_ids_sha256 if self._stable else None,
            "sample_count": self._sample_count if self._stable else None,
            "steps_seen": self._steps_seen,
            "reported_steps": self._reported_steps,
        }

    def report(self):
        state = self.checkpoint_state()
        complete = self._stable and self._steps_seen > 0 and self._reported_steps == self._steps_seen
        if not complete:
            return {
                "status": "unknown",
                "reported_steps": self._reported_steps,
                "completed_steps": self._steps_seen,
                "reason": "stable_sample_ids_not_reported_for_every_completed_step",
            }
        return {
            "status": "reported",
            "algorithm": "ordered-step-sample-ids-sha256-v1",
            "sample_count": state["sample_count"],
            "sample_ids_sha256": state["sample_ids_sha256"],
            "reported_steps": self._reported_steps,
            "completed_steps": self._steps_seen,
            "identity_source": "user_training_context",
        }


def scalar_value(runtime, value, field):
    if isinstance(value, runtime.torch.Tensor):
        if value.numel() != 1:
            raise UserError("STEP_RESULT_INVALID", f"{field} 必须是标量。")
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UserError("STEP_RESULT_INVALID", f"{field} 必须是数值标量。")
    value = float(value)
    if not math.isfinite(value):
        raise UserError("STEP_RESULT_INVALID", f"{field} 包含 NaN 或无穷大。")
    return value


def scalar_map(runtime, value, field):
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > 128:
        raise UserError("STEP_RESULT_INVALID", f"{field} 必须是最多 128 项的对象。")
    result = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name or len(name) > 128:
            raise UserError("STEP_RESULT_INVALID", f"{field} 中的名称必须是 1..128 字符。")
        result[name] = scalar_value(runtime, item, f"{field}.{name}")
    return result


def add_finite_total(totals, counts, name, value, field):
    total = totals.get(name, 0.0) + value
    if not math.isfinite(total):
        raise UserError("STEP_RESULT_INVALID", f"{field}.{name} 累计后超出有限数值范围。")
    totals[name] = total
    counts[name] = counts.get(name, 0) + 1


def callable_checkpoint_hooks(function, label):
    save = getattr(function, "state_dict", None)
    load = getattr(function, "load_state_dict", None)
    if callable(save) != callable(load):
        raise UserError(
            "CHECKPOINT_PROTOCOL_INVALID",
            f"{label} 必须同时提供 state_dict 和 load_state_dict。",
        )
    if callable(save):
        state = save()
        context = callable_checkpoint_context(function, label)
        if context is not None:
            descriptor = context["state_format"]
            _validate_extension_state(state, descriptor, label, saving=True)
            return {
                "kind": "versioned-hooks",
                "state_format": descriptor,
                "callable_identity": context["identity"],
                "state": state,
            }
        return {"kind": "hooks", "state": state}
    if getattr(function, "MODEL_MAVEN_STATELESS", False) is True:
        return {"kind": "stateless"}
    return None


def restore_callable_checkpoint(function, saved, label, allow_reset):
    save = getattr(function, "state_dict", None)
    load = getattr(function, "load_state_dict", None)
    if callable(save) != callable(load):
        raise UserError("CHECKPOINT_PROTOCOL_INVALID", f"{label} 的状态钩子不完整。")
    if saved is None:
        if getattr(function, "MODEL_MAVEN_STATELESS", False) is True:
            return "stateless"
        if not allow_reset:
            raise UserError(
                "RESUME_STATE_UNAVAILABLE",
                f"{label} 没有声明无状态，也没有保存/加载钩子。",
                "实现 state_dict/load_state_dict，或明确使用 --allow-state-reset 接受状态重置。",
            )
        return "reset"
    if not isinstance(saved, dict):
        raise UserError("CHECKPOINT_INVALID", f"{label} checkpoint 状态必须是对象。")
    if saved.get("kind") == "stateless":
        if getattr(function, "MODEL_MAVEN_STATELESS", False) is not True:
            raise UserError("RESUME_STATE_INCOMPATIBLE", f"{label} 的无状态声明已变化。")
        return "stateless"
    if saved.get("kind") == "versioned-hooks":
        if not callable(load):
            if allow_reset:
                return "reset"
            raise UserError("RESUME_STATE_UNAVAILABLE", f"{label} 缺少 load_state_dict 状态钩子。")
        context = callable_checkpoint_context(function, label)
        if context is None:
            if allow_reset:
                return "reset"
            raise UserError(
                "RESUME_STATE_FORMAT_UNAVAILABLE",
                f"{label} 当前版本没有声明 checkpoint 所用的状态格式。",
            )
        saved_format = saved.get("state_format")
        try:
            validate_state_format(saved_format)
        except (TypeError, ValueError) as error:
            raise UserError(
                "CHECKPOINT_STATE_FORMAT_INVALID",
                f"{label} checkpoint 的状态格式声明无效：{error}",
            ) from error
        _validate_extension_state(saved.get("state"), saved_format, label, saving=False)
        current_format = context["state_format"]
        if saved_format["format_id"] != current_format["format_id"]:
            raise UserError(
                "RESUME_STATE_FORMAT_INCOMPATIBLE",
                f"{label} 状态格式 ID 已变化：{saved_format['format_id']} → {current_format['format_id']}。",
                "不能跨不同状态格式猜测迁移；保留旧 checkpoint 或显式重置状态。",
            )
        state = saved["state"]
        if saved_format["version"] != current_format["version"]:
            source_version = _state_version_tuple(saved_format["version"])
            target_version = _state_version_tuple(current_format["version"])
            if target_version < source_version:
                raise UserError(
                    "RESUME_STATE_FORMAT_DOWNGRADE",
                    f"{label} 不支持状态格式降级：{saved_format['version']} → {current_format['version']}。",
                )
            route = f"{saved_format['version']}->{current_format['version']}"
            migrator = context["migrators"].get(route)
            route_manifest = next(
                (item for item in context["migration_manifest"] if item["from_version"] == saved_format["version"] and item["to_version"] == current_format["version"]),
                None,
            )
            if migrator is None or route_manifest is None:
                if allow_reset:
                    return "reset"
                raise UserError(
                    "RESUME_STATE_MIGRATOR_REQUIRED",
                    f"{label} 状态格式需要迁移：{saved_format['version']} → {current_format['version']}。",
                    "为当前 callable 绑定 StateMigrator，或明确使用 --allow-state-reset 接受状态重置。",
                )
            try:
                state = migrator(
                    copy.deepcopy(state),
                    from_version=saved_format["version"],
                    to_version=current_format["version"],
                )
            except Exception as error:
                if allow_reset:
                    return "reset"
                raise UserError(
                    "RESUME_STATE_MIGRATION_FAILED",
                    f"{label} StateMigrator 失败：{type(error).__name__}: {error}",
                    "checkpoint 文件保持不变；修复迁移器或显式重置状态。",
                ) from error
            _validate_extension_state(state, current_format, label, saving=False)
        elif saved_format != current_format:
            raise UserError(
                "RESUME_STATE_FORMAT_DRIFT",
                f"{label} 的状态 Schema 在版本 {current_format['version']} 下发生变化。",
                "给状态格式升版本并提供 StateMigrator；不会按相同版本猜测兼容。",
            )
        try:
            load(state)
        except Exception as error:
            raise UserError(
                "RESUME_STATE_INCOMPATIBLE",
                f"{label} 无法载入已验证的状态：{type(error).__name__}: {error}",
            ) from error
        return "migrated" if saved_format["version"] != current_format["version"] else "restored"
    if saved.get("kind") == "hooks":
        context = callable_checkpoint_context(function, label)
        if context is not None:
            if allow_reset:
                return "reset"
            raise UserError(
                "RESUME_STATE_FORMAT_MISSING",
                f"{label} checkpoint 未绑定当前 callable 声明的状态格式。",
                "该旧 checkpoint 无法安全校验；使用兼容版本或显式重置状态。",
            )
    if saved.get("kind") != "hooks" or not callable(load):
        if not allow_reset:
            raise UserError("RESUME_STATE_UNAVAILABLE", f"{label} 的状态无法恢复。")
        return "reset"
    load(saved["state"])
    return "restored"


def callable_checkpoint_context(function, label):
    """Return the callable's validated version manifest and source owner."""
    names = (
        "MODEL_MAVEN_STATE_FORMAT",
        "MODEL_MAVEN_STATE_VERSION",
        "MODEL_MAVEN_STATE_SCHEMA",
    )
    values = [getattr(function, name, None) for name in names]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise UserError(
            "CHECKPOINT_STATE_FORMAT_INVALID",
            f"{label} 必须同时声明 MODEL_MAVEN_STATE_FORMAT、_VERSION 和 _SCHEMA。",
        )
    descriptor = {
        "format_id": copy.deepcopy(values[0]),
        "version": copy.deepcopy(values[1]),
        "schema": copy.deepcopy(values[2]),
    }
    try:
        validate_state_format(descriptor)
    except (TypeError, ValueError) as error:
        raise UserError(
            "CHECKPOINT_STATE_FORMAT_INVALID",
            f"{label} 状态格式声明无效：{error}",
        ) from error
    raw_migrators = getattr(function, "MODEL_MAVEN_STATE_MIGRATORS", {})
    raw_manifest = getattr(function, "MODEL_MAVEN_STATE_MIGRATION_MANIFEST", [])
    if not isinstance(raw_migrators, dict) or not isinstance(raw_manifest, list):
        raise UserError(
            "CHECKPOINT_STATE_FORMAT_INVALID",
            f"{label} 的 StateMigrator 声明必须是映射和清单数组。",
        )
    migration_manifest = []
    seen_routes = set()
    for entry in raw_manifest:
        if not isinstance(entry, dict) or set(entry) != {"format_id", "from_version", "to_version", "entrypoint"}:
            raise UserError("CHECKPOINT_STATE_FORMAT_INVALID", f"{label} 的 StateMigrator 清单格式无效。")
        from_version, to_version = entry.get("from_version"), entry.get("to_version")
        if (
            entry.get("format_id") != descriptor["format_id"]
            or not isinstance(from_version, str)
            or not isinstance(to_version, str)
            or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", from_version)
            or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", to_version)
            or _state_version_tuple(to_version) <= _state_version_tuple(from_version)
            or to_version != descriptor["version"]
        ):
            raise UserError("CHECKPOINT_STATE_FORMAT_INVALID", f"{label} 的 StateMigrator 版本路由无效。")
        route = f"{entry['from_version']}->{entry['to_version']}"
        migrator = raw_migrators.get(route)
        if (
            not callable(migrator)
            or f"{migrator.__module__}:{migrator.__qualname__}" != entry.get("entrypoint")
            or route in seen_routes
        ):
            raise UserError("CHECKPOINT_STATE_FORMAT_INVALID", f"{label} 的 StateMigrator 与声明清单不匹配。")
        seen_routes.add(route)
        migration_manifest.append(dict(entry))
    if set(raw_migrators) != {f"{item['from_version']}->{item['to_version']}" for item in migration_manifest}:
        raise UserError("CHECKPOINT_STATE_FORMAT_INVALID", f"{label} 含有未声明的 StateMigrator。")
    return {
        "state_format": descriptor,
        "migration_manifest": migration_manifest,
        "migrators": dict(raw_migrators),
        "identity": _callable_source_identity(function),
    }


def _validate_extension_state(state, descriptor, label, saving):
    try:
        validate_state_value(state, descriptor)
    except (TypeError, ValueError) as error:
        code = "CHECKPOINT_STATE_FORMAT_INVALID" if saving else "CHECKPOINT_STATE_INVALID"
        action = "保存" if saving else "恢复"
        raise UserError(
            code,
            f"{label} 的扩展状态不符合声明的 Schema，无法{action}：{error}",
        ) from error


def _state_version_tuple(value):
    return tuple(int(part) for part in value.split("."))


def checkpoint_environment_inventory(snapshot):
    """Return the minimal normalized package inventory used by migrations.

    The source snapshot deliberately excludes private index configuration,
    credentials and requirements text. Ambiguities are retained so migration
    authorization can fail closed.
    """
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        return None
    python = snapshot.get("python")
    distributions = snapshot.get("distributions")
    if not isinstance(python, dict) or not isinstance(distributions, list):
        return None
    required_python = ("implementation", "version", "platform", "machine", "abi")
    if any(not isinstance(python.get(key), str) or not python[key] for key in required_python):
        return None
    normalized = []
    for item in distributions:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        version = item.get("version")
        source_kind = item.get("source_kind")
        if not all(isinstance(value, str) and value for value in (name, version, source_kind)):
            return None
        normalized.append({
            "name": _distribution_name_key(name),
            "version": version,
            "source_kind": source_kind,
        })
    normalized.sort(key=lambda item: (item["name"], item["version"], item["source_kind"]))
    ambiguous = snapshot.get("ambiguous_distribution_names", [])
    issues = snapshot.get("issues", [])
    if not isinstance(ambiguous, list) or not isinstance(issues, list):
        return None
    ambiguous_names = []
    for item in ambiguous:
        if isinstance(item, dict) and isinstance(item.get("normalized_name"), str):
            ambiguous_names.append(_distribution_name_key(item["normalized_name"]))
        elif isinstance(item, str):
            ambiguous_names.append(_distribution_name_key(item))
        else:
            return None
    return {
        "schema_version": 1,
        "python": {key: python[key] for key in required_python},
        "pytorch_version": snapshot.get("pytorch_version"),
        "pytorch_cuda_version": snapshot.get("pytorch_cuda_version"),
        "distributions": normalized,
        "ambiguous_distribution_names": sorted(set(ambiguous_names)),
        "issues": sorted(str(item) for item in issues),
        "fingerprint_sha256": snapshot.get("fingerprint_sha256"),
    }


def _distribution_name_key(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def _callable_source_identity(function):
    module_name = getattr(function, "__module__", None)
    if not isinstance(module_name, str) or not module_name:
        return {"module": None, "source_file": None, "distribution": None}
    module = sys.modules.get(module_name)
    filename = inspect.getsourcefile(function) or getattr(module, "__file__", None)
    if not isinstance(filename, str):
        return {"module": module_name, "source_file": None, "distribution": None}
    try:
        source_file = str(Path(filename).resolve(strict=True))
    except (OSError, ValueError):
        return {"module": module_name, "source_file": None, "distribution": None}
    top_level = module_name.split(".", 1)[0]
    try:
        candidates = importlib.metadata.packages_distributions().get(top_level, [])
    except Exception:
        candidates = []
    names = {_distribution_name_key(name): name for name in candidates if isinstance(name, str) and name}
    if len(names) != 1:
        return {"module": module_name, "source_file": source_file, "distribution": None}
    name_key, distribution_name = next(iter(names.items()))
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        files = distribution.files
        if files is None:
            return {"module": module_name, "source_file": source_file, "distribution": None}
        package_files = {
            str(Path(distribution.locate_file(item)).resolve())
            for item in files
            if str(item).endswith(".py")
        }
        if source_file not in package_files:
            return {"module": module_name, "source_file": source_file, "distribution": None}
        direct_url = distribution.read_text("direct_url.json")
        source_kind = "registry_or_unknown"
        if direct_url:
            try:
                source = json.loads(direct_url)
            except (TypeError, json.JSONDecodeError):
                source_kind = "direct_unknown"
            else:
                if isinstance(source, dict):
                    if source.get("vcs_info"):
                        source_kind = "vcs"
                    elif isinstance(source.get("dir_info"), dict):
                        source_kind = "editable" if source["dir_info"].get("editable") is True else "local"
                    elif str(source.get("url", "")).startswith("file:"):
                        source_kind = "local"
                    else:
                        source_kind = "direct"
        parts = module_name.split(".")
        source_path = Path(source_file)
        if source_path.name == "__init__.py":
            package_root = source_path.parent
            for _ in range(max(0, len(parts) - 1)):
                package_root = package_root.parent
        elif len(parts) == 1:
            package_root = source_path
        else:
            package_root = source_path.parent
            for _ in range(max(0, len(parts) - 2)):
                package_root = package_root.parent
        return {
            "module": module_name,
            "source_file": source_file,
            "distribution": {
                "name": name_key,
                "version": str(distribution.version),
                "source_kind": source_kind,
                "package_root": str(package_root.resolve()),
            },
        }
    except Exception:
        return {"module": module_name, "source_file": source_file, "distribution": None}


def authorize_checkpoint_state_migration(payload, request, source_manifest):
    """Authorize exactly one callable's declared state upgrade across a resume.

    All model/data/other training sources, Python ABI, PyTorch build, and all
    other distributions must remain byte-for-byte equivalent in their
    normalized identity. The extension distribution must be uniquely mapped
    to the stateful callable and its source changes must stay within that
    package's recorded root.
    """
    state_slots = ("batch_state", "step_state", "loop_state")
    versioned = [
        (slot.removesuffix("_state"), payload.get(slot))
        for slot in state_slots
        if isinstance(payload.get(slot), dict) and payload[slot].get("kind") == "versioned-hooks"
    ]
    if len(versioned) != 1:
        return None
    slot, saved_state = versioned[0]
    contexts = request.get("_checkpoint_callable_contexts")
    if not isinstance(contexts, dict):
        return None
    context = contexts.get(slot)
    if not isinstance(context, dict):
        return None
    saved_format = saved_state.get("state_format")
    current_format = context.get("state_format")
    if not isinstance(saved_format, dict) or not isinstance(current_format, dict):
        return None
    if saved_format.get("format_id") != current_format.get("format_id"):
        return None
    route = f"{saved_format.get('version')}->{current_format.get('version')}"
    exact_routes = [
        item for item in context.get("migration_manifest", [])
        if isinstance(item, dict)
        and item.get("format_id") == current_format.get("format_id")
        and item.get("from_version") == saved_format.get("version")
        and item.get("to_version") == current_format.get("version")
    ]
    if route not in context.get("migrators", {}) or len(exact_routes) != 1:
        return None
    old_owner = saved_state.get("callable_identity")
    current_owner = context.get("identity")
    if not isinstance(old_owner, dict) or not isinstance(current_owner, dict):
        return None
    old_distribution = old_owner.get("distribution")
    current_distribution = current_owner.get("distribution")
    if not isinstance(old_distribution, dict) or not isinstance(current_distribution, dict):
        return None
    if (
        old_owner.get("module") != current_owner.get("module")
        or old_distribution.get("name") != current_distribution.get("name")
        or old_distribution.get("source_kind") != current_distribution.get("source_kind")
        or old_distribution.get("version") == current_distribution.get("version")
    ):
        return None
    old_inventory = payload.get("environment_inventory")
    current_inventory = request.get("_checkpoint_environment_inventory")
    if not _same_except_extension_distribution(old_inventory, current_inventory, old_distribution, current_distribution):
        return None
    old_root = old_distribution.get("package_root")
    current_root = current_distribution.get("package_root")
    if not _source_changes_within_roots(payload.get("source_manifest"), source_manifest, (old_root, current_root)):
        return None
    return {
        "slot": slot,
        "format_id": current_format["format_id"],
        "from_version": saved_format["version"],
        "to_version": current_format["version"],
        "distribution": current_distribution["name"],
    }


def _same_except_extension_distribution(old, current, old_owner, current_owner):
    if not isinstance(old, dict) or not isinstance(current, dict):
        return False
    for key in ("python", "pytorch_version", "pytorch_cuda_version"):
        if old.get(key) != current.get(key):
            return False
    old_ambiguous = old.get("ambiguous_distribution_names")
    current_ambiguous = current.get("ambiguous_distribution_names")
    if (
        not isinstance(old_ambiguous, list)
        or old_ambiguous != current_ambiguous
        or old_owner.get("name") in old_ambiguous
    ):
        return False
    if old.get("issues") or current.get("issues"):
        return False
    if not isinstance(old.get("fingerprint_sha256"), str) or not isinstance(current.get("fingerprint_sha256"), str):
        return False
    def entries(inventory):
        result = []
        for item in inventory.get("distributions", []):
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                return None
            if not isinstance(item.get("version"), str) or not isinstance(item.get("source_kind"), str):
                return None
            result.append((item["name"], item["version"], item["source_kind"]))
        return result
    old_entries = entries(old)
    current_entries = entries(current)
    if old_entries is None or current_entries is None:
        return False
    expected_name = old_owner.get("name")
    if expected_name != current_owner.get("name"):
        return False
    before_owner = (expected_name, old_owner.get("version"), old_owner.get("source_kind"))
    after_owner = (expected_name, current_owner.get("version"), current_owner.get("source_kind"))
    old_owner_entries = [entry for entry in old_entries if entry[0] == expected_name]
    current_owner_entries = [entry for entry in current_entries if entry[0] == expected_name]
    if old_owner_entries != [before_owner] or current_owner_entries != [after_owner]:
        return False
    old_other = Counter(entry for entry in old_entries if entry[0] != expected_name)
    current_other = Counter(entry for entry in current_entries if entry[0] != expected_name)
    return old_other == current_other


def _source_changes_within_roots(old, current, roots):
    if not isinstance(old, dict) or not isinstance(current, dict):
        return False
    if old == current:
        return True
    changed = {
        path for path in old.keys() | current.keys()
        if old.get(path) != current.get(path)
    }
    valid_roots = []
    for value in roots:
        if isinstance(value, str) and value:
            try:
                valid_roots.append(Path(value).resolve())
            except (OSError, ValueError):
                return False
    if not valid_roots or not changed:
        return False
    for raw_path in changed:
        try:
            path = Path(raw_path).resolve()
        except (OSError, ValueError):
            return False
        if not any(path == root or root in path.parents for root in valid_roots):
            return False
    return True


def checkpoint_rng_state(runtime):
    torch_cuda = []
    cuda = getattr(runtime.torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        try:
            torch_cuda = cuda.get_rng_state_all()
        except Exception as error:
            raise UserError(
                "CHECKPOINT_DEVICE_RNG_UNAVAILABLE",
                f"无法读取 CUDA 设备随机状态：{error}",
                "检查设备是否仍可用；为保证可复现性，本次不提交不完整 checkpoint。",
            ) from error
    return {
        "python": runtime.python_random.getstate(),
        "torch_cpu": runtime.torch.get_rng_state(),
        "torch_cuda": torch_cuda,
    }


def restore_rng_state(runtime, state):
    if not isinstance(state, dict) or "python" not in state or "torch_cpu" not in state:
        raise UserError("CHECKPOINT_INVALID", "checkpoint 缺少 Python/Torch RNG 状态。")
    cuda = getattr(runtime.torch, "cuda", None)
    cuda_available = cuda is not None and cuda.is_available()
    saved_cuda = state.get("torch_cuda")
    if saved_cuda is None:
        if cuda_available:
            raise UserError(
                "RESUME_DEVICE_RNG_UNAVAILABLE",
                "checkpoint 缺少当前 CUDA 设备的随机状态。",
                "使用创建该 checkpoint 的设备状态恢复，或显式接受状态重置后重新开始。",
            )
        saved_cuda = []
    if not isinstance(saved_cuda, list):
        raise UserError("CHECKPOINT_INVALID", "checkpoint CUDA RNG 状态必须是数组。")
    if saved_cuda and not cuda_available:
        raise UserError(
            "RESUME_DEVICE_INCOMPATIBLE",
            "checkpoint 包含 CUDA 随机状态，但当前运行环境没有可用 CUDA 设备。",
            "恢复原设备环境后重试。",
        )
    if cuda_available:
        current_count = cuda.device_count()
        if len(saved_cuda) != current_count:
            raise UserError(
                "RESUME_DEVICE_INCOMPATIBLE",
                f"checkpoint 保存了 {len(saved_cuda)} 个 CUDA 设备状态，当前环境可见 {current_count} 个。",
                "恢复到相同设备数量的环境后重试。",
            )
    runtime.python_random.setstate(state["python"])
    runtime.torch.set_rng_state(state["torch_cpu"])
    if saved_cuda:
        try:
            cuda.set_rng_state_all(saved_cuda)
        except Exception as error:
            raise UserError(
                "RESUME_DEVICE_RNG_INVALID",
                f"无法恢复 CUDA 设备随机状态：{error}",
                "检查设备与 PyTorch 版本是否匹配；checkpoint 保持不变。",
            ) from error
