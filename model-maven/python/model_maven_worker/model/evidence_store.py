"""Atomic metadata-only storage for model probe evidence with strict freshness keys."""

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import tempfile
from pathlib import Path


_FORMAT = 2
_IGNORED_DIRS = {
    ".git", ".model-maven", "__pycache__", "node_modules", ".pytest_cache",
}
_SOURCE_SUFFIXES = {".py", ".toml", ".yaml", ".yml"}
_MAX_SOURCE_FILES = 20000
_MAX_SOURCE_DIRS = 20000
_MAX_SOURCE_ENTRIES = 200000
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_MAX_WEIGHT_BYTES = 512 * 1024 * 1024
_MAX_CACHE_RECORD_BYTES = 32 * 1024 * 1024
_MAX_CACHE_RECORDS = 64


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _value_digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _source_fingerprint(project_root, model, import_roots=None):
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    paths = set()
    source_inventory = {}
    total_bytes = 0
    try:
        if import_roots is not None and (not isinstance(import_roots, list) or len(import_roots) > 32):
            return None, None, "project import roots are invalid"
        selected = []
        for value in import_roots or []:
            if (not isinstance(value, str) or not value or "\\" in value
                    or any(part in {"", ".", ".."} for part in value.split("/"))):
                return None, None, "project import roots are invalid"
            relative = Path(value)
            if relative.is_absolute() or any(part in {".git", ".model-maven", "__pycache__"} for part in relative.parts):
                return None, None, "project import roots cannot be fingerprinted"
            candidate = root / relative
            if not candidate.is_dir() or candidate.resolve() != candidate:
                return None, None, "project import root is unavailable or uses a symlink"
            selected.append(candidate)

        def selected_path(path):
            return any(path == chosen or path in chosen.parents or chosen in path.parents for chosen in selected)

        environment = Path(sys.prefix).resolve()
        excluded_roots = ({environment} if environment != root and environment.is_relative_to(root) else set())
        def include_directory(path):
            if path.name in {".git", ".model-maven", "__pycache__"} or path.is_symlink():
                return False
            if (path.name in _IGNORED_DIRS or (path / "pyvenv.cfg").is_file()
                    or (path.name == "target" and (path / ".rustc_info.json").is_file())):
                excluded_roots.add(path)
            if selected_path(path):
                return True
            if path in excluded_roots or any(parent in excluded_roots for parent in path.parents):
                return False
            return True

        directory_count = 0
        entry_count = 0
        def raise_walk_error(error):
            raise error

        for current, dirs, files in os.walk(root, followlinks=False, onerror=raise_walk_error):
            directory_count += 1
            entry_count += len(dirs) + len(files)
            if directory_count > _MAX_SOURCE_DIRS or entry_count > _MAX_SOURCE_ENTRIES:
                return None, None, "project directory inventory exceeds the cache verification limit"
            dirs[:] = sorted(name for name in dirs if include_directory(Path(current) / name))
            for name in files:
                path = Path(current) / name
                if not selected_path(path) and any(parent in excluded_roots for parent in path.parents):
                    continue
                if path.suffix.lower() in _SOURCE_SUFFIXES and not path.is_symlink():
                    if path.suffix.lower() in {".yaml", ".yml"} and path.stat().st_size > 1024 * 1024:
                        return None, None, "large YAML source could not be safely fingerprinted"
                    paths.add(path)
                    if len(paths) > _MAX_SOURCE_FILES:
                        return None, None, "project source inventory exceeds the cache verification limit"
        project_paths = set(paths)
        for loaded in list(sys.modules.values()):
            filename = getattr(loaded, "__file__", None)
            if not isinstance(filename, str):
                continue
            source = Path(filename).resolve()
            if source.suffix != ".py" or not source.is_relative_to(root):
                continue
            if environment != root and environment.is_relative_to(root) and source.is_relative_to(environment) and not selected_path(source):
                continue
            if source not in project_paths:
                return None, None, "an observed project module is outside the fingerprinted tree"
        for module in model.modules():
            module_name = type(module).__module__
            top_level = module_name.split(".", 1)[0]
            if top_level in {"torch", "builtins"}:
                continue
            try:
                source = inspect.getsourcefile(type(module))
            except (OSError, TypeError):
                source = None
            if not source:
                return None, None, f"module source is unavailable for {module_name}"
            source_path = Path(source).resolve()
            try:
                source_path.relative_to(root)
            except ValueError:
                if not source_path.is_file() or source_path.is_symlink():
                    return None, None, f"external module source is unavailable for {module_name}"
                paths.add(source_path)
            else:
                if source_path not in project_paths:
                    return None, None, f"model source is outside the fingerprinted project tree: {module_name}"
        if len(paths) > _MAX_SOURCE_FILES:
            return None, None, "model dependency source inventory exceeds the cache verification limit"
        for path in sorted(paths):
            try:
                relative_text = path.relative_to(root).as_posix()
            except ValueError:
                relative_text = "external:" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()
            relative = relative_text.encode("utf-8")
            content_digest = hashlib.sha256()
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                total_bytes += before.st_size
                if total_bytes > _MAX_SOURCE_BYTES:
                    return None, None, "project source files exceed the cache verification limit"
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(before.st_size.to_bytes(8, "big"))
                count = 0
                while chunk := stream.read(1024 * 1024):
                    count += len(chunk)
                    if count > before.st_size:
                        return None, None, "project source changed while fingerprinting"
                    digest.update(chunk)
                    content_digest.update(chunk)
                after = os.fstat(stream.fileno())
            current = path.stat()
            identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            if count != before.st_size or identity(before) != identity(after) or identity(after) != identity(current):
                return None, None, "project source changed while fingerprinting"
            source_inventory[hashlib.sha256(relative).hexdigest()] = content_digest.hexdigest()
    except (OSError, ValueError) as error:
        return None, None, f"project source inventory is incomplete: {type(error).__name__}"
    if not paths:
        return None, None, "no project source files were found"
    return digest.hexdigest(), source_inventory, None


def _weights_fingerprint(model, torch_module):
    digest = hashlib.sha256()
    total = 0
    try:
        for name, tensor in sorted(model.state_dict().items()):
            if not isinstance(tensor, torch_module.Tensor):
                return None, "model state contains non-Tensor values"
            size = tensor.numel() * tensor.element_size()
            if size > _MAX_WEIGHT_BYTES - total:
                return None, "model weights exceed the cache verification limit"
            value = tensor.detach()
            if value.is_sparse:
                value = value.to_dense()
            value = value.cpu().contiguous()
            raw = memoryview(value.reshape(-1).view(torch_module.uint8).numpy())
            total += len(raw)
            key = name.encode("utf-8")
            digest.update(len(key).to_bytes(8, "big"))
            digest.update(key)
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(_canonical(list(value.shape)).encode("ascii"))
            digest.update(raw)
    except Exception as error:
        return None, f"model weight fingerprint failed: {type(error).__name__}"
    return digest.hexdigest(), None


def weight_fingerprint_report(model, torch_module):
    """Describe the current registered state without exposing tensor values."""
    digest, reason = _weights_fingerprint(model, torch_module)
    return {
        "status": "available" if digest is not None else "unknown",
        "algorithm": "sha256-registered-state-v1" if digest is not None else None,
        "digest": digest,
        "scope": "model.state_dict() tensor names, dtypes, shapes and bytes",
        "reason": reason,
        "covers": "registered state_dict tensors only; not unregistered state, external resources or future mutations",
    }


def environment_fingerprint(torch_module):
    """Return a privacy-bounded runtime identity and explicit scope limits."""
    package_versions = []
    try:
        for distribution in importlib.metadata.distributions():
            package_versions.append([
                distribution.metadata.get("Name", "").lower(), distribution.version
            ])
        package_versions.sort()
    except Exception as error:
        package_versions = None
        package_reason = type(error).__name__
    identity = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "torch": str(torch_module.__version__),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": package_versions,
    }
    return {
        "status": "available" if package_versions is not None else "partial",
        "python": identity["python"],
        "python_implementation": identity["python_implementation"],
        "torch": identity["torch"],
        "platform_system": identity["platform_system"],
        "platform_machine": identity["platform_machine"],
        "package_count": len(package_versions) if package_versions is not None else None,
        "package_inventory_sha256": _value_digest(package_versions) if package_versions is not None else None,
        "identity_sha256": _value_digest(identity) if package_versions is not None else None,
        "package_inventory_reason": None if package_versions is not None else package_reason,
        "scope": "runtime versions and installed distribution metadata; excludes executable path, environment variables, native libraries and device driver state",
    }


def build_evidence_identity(
    model, request, *, torch_module, input_content=None, contract=None, source_inventory_out=None,
    weights_report=None,
):
    """Return a complete reusable identity, or a reason to run a fresh probe."""
    observation = request.get("_data_observation")
    sampling = observation.get("sampling", {}) if isinstance(observation, dict) else {}
    if not (
        isinstance(observation, dict)
        and isinstance(observation.get("dataset_fingerprint"), str)
        and observation["dataset_fingerprint"]
        and isinstance(sampling.get("sample_ids"), list)
        and sampling["sample_ids"]
        and sampling.get("stable_sample_ids") is True
    ):
        return None, "batch has no stable dataset fingerprint and sample IDs"
    if request.get("training"):
        return None, "training-mode probes may depend on mutable module or RNG state"
    if request.get("seed") is None and request.get("variant") is None:
        return None, "ambient RNG state is not fixed for this probe"
    if not (isinstance(input_content, dict) and input_content.get("status") == "available"
            and isinstance(input_content.get("algorithm"), str)
            and isinstance(input_content.get("digest"), str)):
        return None, "forward input content cannot be completely fingerprinted"
    # User code can change cwd and resolve relative paths against it. Retain
    # only digests in persistent cache keys, and fail closed when unknown.
    try:
        current_working_directory = os.getcwd()
    except OSError:
        return None, "working directory is unavailable"
    before_model = request.get("_working_directory_before_model")
    after_model = request.get("_working_directory_after_model")
    if not isinstance(before_model, str) or not isinstance(after_model, str):
        return None, "working directory at model construction is unknown"
    source_hash, source_inventory, reason = _source_fingerprint(
        request.get("project_root", "."), model,
        request.get("runtime", {}).get("project_import_roots")
    )
    if reason:
        return None, reason
    if isinstance(source_inventory_out, dict):
        source_inventory_out.clear()
        source_inventory_out.update(source_inventory)
    if not isinstance(weights_report, dict):
        weights_report = weight_fingerprint_report(model, torch_module)
    weights_hash = weights_report.get("digest")
    if not isinstance(weights_hash, str):
        return None, weights_report.get("reason") or "model weights cannot be fingerprinted"
    identity = {
        "format": _FORMAT,
        "source_sha256": source_hash,
        "weights_sha256": weights_hash,
        "batch": {
            "dataset_fingerprint_sha256": _value_digest(observation["dataset_fingerprint"]),
            "source_sha256": _value_digest(observation.get("source")),
            "split_sha256": _value_digest(observation.get("split")),
            "sampling_algorithm": sampling.get("algorithm"),
            "sampling_seed": sampling.get("seed"),
            "sample_count": len(sampling["sample_ids"]),
            "sample_ids_sha256": _value_digest(sampling["sample_ids"]),
            "input_content_algorithm": input_content["algorithm"],
            "input_content_sha256": input_content["digest"],
        },
        "model_sha256": _value_digest(request.get("model")),
        "model_params_sha256": _value_digest(request.get("model_params", {})),
        "model_injections_sha256": _value_digest(request.get("model_injections", {})),
        "batch_factory_sha256": _value_digest(request.get("batch")),
        "batch_params_sha256": _value_digest(request.get("batch_params", {})),
        "batch_binding": request.get("batch_binding"),
        "requested_batch_fingerprint_sha256": _value_digest(request.get("data_fingerprint")),
        "variant_sha256": _value_digest(request.get("variant")),
        "contract_sha256": _value_digest(contract),
        "contract_content_sha256": request.get("_contract_content_sha256"),
        "contract_identity_required": request.get("if_contract_sha256") is not None,
        "protocol": {
            "protocol_version": request.get("protocol_version"),
            "worker_api_version": request.get("worker_api_version", 1),
            # Version 8 captures selected subtrees independently of the
            # bounded full-model trace; older records can omit late calls.
            "evidence_schema": 8,
        },
        "execution": {
            "training": False,
            "project_import_roots_sha256": _value_digest(request.get("runtime", {}).get("project_import_roots")),
            "probe_device_sha256": _value_digest(request.get("_probe_device_report")),
            "working_directory_before_model_sha256": _value_digest(before_model),
            "working_directory_after_model_sha256": _value_digest(after_model),
            "working_directory_after_input_sha256": _value_digest(current_working_directory),
            "seed": request.get("seed"),
            "python": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "torch": str(torch_module.__version__),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "packages": sorted([
                [distribution.metadata.get("Name", "").lower(), distribution.version]
                for distribution in importlib.metadata.distributions()
            ]),
        },
        "selection": {"path": request.get("path"), "call": request.get("call", "all")},
    }
    return identity, None


def evidence_key(identity):
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def _safe_tree(value):
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "tensor":
            return {
                key: value.get(key)
                for key in ("kind", "shape", "dtype", "device", "layout", "memory_format", "requires_grad")
                if key in value
            }
        if kind == "dict":
            return {"kind": "dict", "items": {key: _safe_tree(item) for key, item in value.get("items", {}).items()},
                    "items_truncated": value.get("items_truncated"),
                    "keys_all_strings": value.get("keys_all_strings"),
                    "observed_keys_all_strings": value.get("observed_keys_all_strings")}
        if kind in {"list", "tuple"}:
            return {"kind": kind, "items": [_safe_tree(item) for item in value.get("items", [])],
                    "truncated": value.get("truncated", 0)}
        if kind in {"opaque", "truncated"}:
            return {"kind": kind, "type": value.get("type")}
        return {"kind": kind or "dict"}
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool"}
    if isinstance(value, int):
        return {"kind": "int"}
    if isinstance(value, float):
        return {"kind": "float"}
    if isinstance(value, str):
        return {"kind": "str", "length": min(len(value), 1000000)}
    return {"kind": type(value).__name__}


def _safe_evidence(data):
    calls = []
    for call in data.get("evidence_calls", data.get("calls", [])):
        calls.append({
            key: call[key]
            for key in (
                "call_id", "invocation_id", "probe_id", "forward_id", "batch_id",
                "module_instance_id", "paths", "alias_paths", "module_type", "invocation_index",
                "parent_invocation_id", "input_signature", "input_output_paths",
                "branch_cache_key", "branch_cache_source_path",
            ) if key in call
        } | {
            "inputs": _safe_tree(call.get("inputs")),
            "outputs": _safe_tree(call.get("outputs")),
        })
    shape = data.get("shape_inference")
    safe_shape = None
    if isinstance(shape, dict):
        safe_shape = {key: shape[key] for key in ("status", "engine", "nodes", "diagnostics", "coverage") if key in shape}
    return {
        key: data[key]
        for key in ("probe_id", "forward_id", "batch_id", "batch_identity_stable", "input_signature", "output_signature",
                    "calls_observed", "trace_truncated", "observed_mode", "coverage", "run_provenance")
        if key in data
    } | {
        "inputs": _safe_tree(data.get("inputs")),
        "outputs": _safe_tree(data.get("outputs")),
        "calls": calls,
        "shape_inference": safe_shape,
        "contract_report": data.get("contract_report"),
    }


def load_evidence(project_root, identity):
    key = evidence_key(identity)
    path = Path(project_root) / ".model-maven" / "cache" / "model-evidence" / f"{key}.json"
    try:
        if path.stat().st_size > _MAX_CACHE_RECORD_BYTES:
            return None, key
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, key
    if record.get("format") != _FORMAT or record.get("identity") != identity:
        return None, key
    return record.get("data"), key


def load_evidence_with_inventory(project_root, identity):
    key = evidence_key(identity)
    path = Path(project_root) / ".model-maven" / "cache" / "model-evidence" / f"{key}.json"
    try:
        if path.stat().st_size > _MAX_CACHE_RECORD_BYTES:
            return None, key, None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, key, None
    if record.get("format") != _FORMAT or record.get("identity") != identity:
        return None, key, None
    inventory = record.get("source_inventory")
    if not isinstance(inventory, dict):
        inventory = None
    return record.get("data"), key, inventory


def load_latest_identity(project_root):
    path = Path(project_root) / ".model-maven" / "cache" / "model-evidence" / "latest.json"
    try:
        if path.stat().st_size > 64 * 1024:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record.get("identity") if record.get("format") == _FORMAT else None


def invalidation_reasons(previous, current):
    if not isinstance(previous, dict):
        return []
    fields = {
        "source_sha256": "project_or_component_source_changed",
        "weights_sha256": "model_weights_changed",
        "batch": "dataset_sample_or_input_content_changed",
        "execution": "runtime_dependency_or_seed_changed",
        "model_sha256": "model_entrypoint_changed",
        "model_params_sha256": "model_parameters_changed",
        "model_injections_sha256": "component_injections_changed",
        "batch_factory_sha256": "batch_factory_changed",
        "batch_params_sha256": "batch_parameters_changed",
        "batch_binding": "batch_binding_changed",
        "requested_batch_fingerprint_sha256": "declared_dataset_fingerprint_changed",
        "variant_sha256": "model_variant_changed",
        "contract_sha256": "model_contract_changed",
        "contract_content_sha256": "model_contract_changed",
        "contract_identity_required": "model_contract_confirmation_changed",
        "selection": "probe_scope_changed",
        "protocol": "worker_protocol_changed",
    }
    reasons = [reason for field, reason in fields.items() if previous.get(field) != current.get(field)]
    return reasons


def _non_source_identity(identity):
    return {
        key: value for key, value in identity.items()
        if key != "source_sha256"
    }


def _branch_cache_key(identity, call):
    paths = call.get("paths")
    implementation = call.get("implementation")
    input_content = call.get("input_content_identity")
    output_signature = call.get("output_signature")
    if (
        not isinstance(paths, list)
        or len(paths) != 1
        or not isinstance(paths[0], str)
        or not isinstance(implementation, dict)
        or not isinstance(implementation.get("source_file"), str)
        or not isinstance(implementation.get("source_sha256"), str)
        or not isinstance(input_content, dict)
        or input_content.get("status") != "available"
        or not isinstance(input_content.get("digest"), str)
        or not isinstance(output_signature, str)
        or type(call.get("invocation_index")) is not int
    ):
        return None, None
    source_path_hash = hashlib.sha256(
        implementation["source_file"].encode("utf-8")
    ).hexdigest()
    identity_value = {
        "non_source_identity": _non_source_identity(identity),
        "path": paths[0],
        "module_type": call.get("module_type"),
        "invocation_index": call["invocation_index"],
        "source_path_hash": source_path_hash,
        "source_sha256": implementation["source_sha256"],
        "input_signature": call.get("input_signature"),
        "input_content_sha256": input_content["digest"],
        "output_signature": output_signature,
    }
    return _value_digest(identity_value), source_path_hash


def reconcile_branch_evidence(
    previous_data,
    previous_inventory,
    previous_identity,
    current_identity,
    current_inventory,
    calls,
    *,
    shape_inference,
    trace_truncated,
    repeated_module_instances,
):
    """Reuse only metadata for independently attributable, exactly matching invocations.

    The root forward has already run. This never reuses tensors or skips the current
    forward; it decides which bounded structural evidence can be carried forward.
    """
    report = {
        "status": "whole_probe",
        "reused": [],
        "revalidated": [],
        "root_forward_executed": True,
        "tensor_reuse": False,
    }

    def whole(reason):
        report["expanded_scope_reason"] = reason
        report["revalidated"] = [
            {"path": paths[0] if isinstance(paths, list) and paths else "model",
             "invocation_index": call.get("invocation_index")}
            for call in calls
            for paths in [call.get("paths")]
        ]
        return report

    if not isinstance(shape_inference, dict) or shape_inference.get("status") != "inferred-static":
        return whole("FX/Meta 未证明稳定静态图；动态或不支持路径无法安全局部化。")
    if trace_truncated:
        return whole("模块调用轨迹被截断，受影响范围未知。")
    if repeated_module_instances:
        return whole("存在同一模块实例的重复调用或共享别名，无法确定独立分支边界。")
    if not isinstance(current_inventory, dict):
        return whole("源码文件指纹清单缺失，无法归属变化来源。")

    for call in calls:
        key, source_path_hash = _branch_cache_key(current_identity, call)
        if key is None:
            return whole("调用输入内容指纹不完整或模块身份不稳定；扩大为整模型重验。")
        call["branch_cache_key"] = key
        call["branch_cache_source_path"] = source_path_hash
    if not isinstance(previous_data, dict) or not isinstance(previous_identity, dict):
        return whole("没有可与本次相同配置和稳定样本身份比较的旧证据；执行完整模型探测。")
    changes = invalidation_reasons(previous_identity, current_identity)
    source_reason = "project_or_component_source_changed"
    if source_reason not in changes:
        return whole("没有仅由源码变化触发的增量候选；执行完整模型探测。")
    non_source_changes = [reason for reason in changes if reason != source_reason]
    if non_source_changes:
        return whole(
            "样本、模型权重、环境、契约或探测范围也发生变化（"
            + ", ".join(non_source_changes)
            + "），不能局部复用。"
        )
    if not isinstance(previous_inventory, dict):
        return whole("旧源码文件指纹清单缺失，无法归属变化来源。")

    changed_paths = {
        path_hash
        for path_hash in set(previous_inventory) | set(current_inventory)
        if previous_inventory.get(path_hash) != current_inventory.get(path_hash)
    }
    if not changed_paths:
        return whole("总源码指纹变化未能定位到具体源码文件。")

    calls_by_source = {}
    for call in calls:
        implementation = call.get("implementation")
        source_file = implementation.get("source_file") if isinstance(implementation, dict) else None
        source_sha = implementation.get("source_sha256") if isinstance(implementation, dict) else None
        paths = call.get("paths")
        if not isinstance(source_file, str) or not isinstance(source_sha, str):
            return whole("存在无法取得源码指纹的调用，依赖范围未知。")
        if not isinstance(paths, list) or len(paths) != 1:
            return whole("存在模块别名或多路径共享注册，无法局部归属调用。")
        source_path_hash = hashlib.sha256(source_file.encode("utf-8")).hexdigest()
        calls_by_source.setdefault(source_path_hash, []).append(call)

    for changed_path in changed_paths:
        affected = calls_by_source.get(changed_path)
        if not affected:
            return whole("源码变化无法归属到已观察模块实现文件（可能是模型入口、批次工厂或间接 helper）；扩大为整模型重验。")
        if any(call.get("paths") == ["model"] for call in affected):
            return whole("根模型 forward 源码变化可能改变分支接线；扩大为整模型重验。")

    current_by_id = {
        call.get("invocation_id"): call
        for call in calls
        if isinstance(call.get("invocation_id"), str)
    }
    invalidated_invocations = set()
    for changed_path in changed_paths:
        for call in calls_by_source[changed_path]:
            parent_id = call.get("invocation_id")
            while parent_id and parent_id not in invalidated_invocations:
                invalidated_invocations.add(parent_id)
                parent = current_by_id.get(parent_id)
                parent_id = parent.get("parent_invocation_id") if parent else None

    previous_calls = previous_data.get("calls")
    if not isinstance(previous_calls, list):
        return whole("旧证据缺少完整逐调用记录。")
    previous_by_location = {}
    for call in previous_calls:
        paths = call.get("paths")
        if isinstance(paths, list) and len(paths) == 1:
            location = (paths[0], call.get("module_type"), call.get("invocation_index"))
            previous_by_location.setdefault(location, []).append(call)
    if any(len(items) != 1 for items in previous_by_location.values()):
        return whole("旧调用记录存在歧义，无法一一匹配模块分支。")

    for call in calls:
        key = call["branch_cache_key"]
        paths = call["paths"]
        location = (paths[0], call.get("module_type"), call.get("invocation_index"))
        old = previous_by_location.get(location, [])
        item = {"path": paths[0], "invocation_index": call["invocation_index"]}
        if (
            call.get("invocation_id") not in invalidated_invocations
            and len(old) == 1
            and old[0].get("branch_cache_key") == key
        ):
            report["reused"].append(item)
        else:
            report["revalidated"].append(item)
    if report["reused"]:
        report["status"] = "partial_reuse"
        report["expanded_scope_reason"] = None
    else:
        report["status"] = "whole_probe"
        report["expanded_scope_reason"] = "源码变化影响了所有可归属模块调用；整模型证据均已重验。"
    return report


def _atomic_write(path, payload):
    fd, temporary = tempfile.mkstemp(prefix=".model-evidence-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def save_evidence(project_root, identity, data, *, source_inventory=None):
    if data.get("contract_report", {}).get("status") == "failed":
        return None
    key = evidence_key(identity)
    directory = Path(project_root) / ".model-maven" / "cache" / "model-evidence"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    path = directory / f"{key}.json"
    record = {
        "format": _FORMAT,
        "identity": identity,
        "source_inventory": source_inventory if isinstance(source_inventory, dict) else None,
        "data": _safe_evidence(data),
    }
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _atomic_write(path, payload)
    latest = {"format": _FORMAT, "identity": identity}
    _atomic_write(directory / "latest.json", json.dumps(latest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    records = sorted(
        (entry for entry in directory.glob("*.json") if entry.is_file() and entry.name != "latest.json"),
        key=lambda entry: entry.stat().st_mtime_ns,
        reverse=True,
    )
    for stale in records[_MAX_CACHE_RECORDS:]:
        try:
            stale.unlink()
        except OSError:
            pass
    return key
