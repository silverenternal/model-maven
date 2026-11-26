"""Real-batch forward tracing and gradient verification."""

import ast
import hashlib
import hmac
import inspect
import json
import math
import re
import sys
import sysconfig
import threading
import traceback
import uuid
from pathlib import Path

from ..protocol import UserError
from ..tensor_inputs.content_identity import fingerprint as _fingerprint_forward_inputs
from ..tensor_inputs import snapshot as input_snapshot
from .contracts import check_contract, load_contract
from .evidence_store import (
    build_evidence_identity,
    environment_fingerprint,
    invalidation_reasons,
    load_evidence,
    load_evidence_with_inventory,
    load_latest_identity,
    reconcile_branch_evidence,
    save_evidence,
    weight_fingerprint_report,
)
from .local_probe import replay_invocation
from .identity_links import TensorIdentityTrace
from .shape import infer_with_fx_meta


_MAX_BRANCH_FINGERPRINT_BYTES = 64 * 1024 * 1024


def _declaration_digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _safe_data_observation(observation):
    """Retain reproducibility metadata while replacing user identifiers with digests."""
    if not isinstance(observation, dict):
        return None
    sampling = observation.get("sampling", {})
    if not isinstance(sampling, dict):
        sampling = {}
    sample_ids = sampling.get("sample_ids")
    return {
        "source_sha256": _declaration_digest(observation.get("source"))
        if observation.get("source") is not None else None,
        "adapter": observation.get("adapter") if isinstance(observation.get("adapter"), str) else None,
        "adapter_version": observation.get("adapter_version")
        if isinstance(observation.get("adapter_version"), str) else None,
        "split_sha256": _declaration_digest(observation.get("split"))
        if observation.get("split") is not None else None,
        "dataset_fingerprint_sha256": _declaration_digest(observation.get("dataset_fingerprint"))
        if observation.get("dataset_fingerprint") is not None else None,
        "sampling": {
            key: sampling.get(key)
            for key in ("algorithm", "seed_decimal", "requested_samples", "actual_samples",
                        "stable_sample_ids", "seed_source")
            if isinstance(sampling.get(key), (str, int, bool)) or sampling.get(key) is None
        } | {
            "sample_count": len(sample_ids) if isinstance(sample_ids, list) else None,
            "sample_ids_sha256": _declaration_digest(sample_ids)
            if isinstance(sample_ids, list) and sample_ids else None,
        },
    }


def _input_source_compatibility(provenance):
    """Expose the bounded source kind through the existing TUI input-source key."""
    input_report = provenance.get("input") if isinstance(provenance, dict) else None
    if not isinstance(input_report, dict):
        return {"batch": {"kind": "unknown", "checkability": "unknown"}}
    return {
        "batch": {
            "kind": input_report.get("source_kind", "unknown"),
            "checkability": input_report.get("checkability", "unknown"),
        }
    }


def _run_provenance(model, request, *, torch_module, batch_content, weight_report,
                    execution_status, calls=None, trace_truncated=False,
                    snapshot_status=None, calls_observed=None,
                    post_weight_report=None, output_signature=None):
    observation = request.get("_data_observation")
    sampling = observation.get("sampling", {}) if isinstance(observation, dict) else {}
    if not isinstance(sampling, dict):
        sampling = {}
    variant = request.get("variant")
    execution_seed = request.get("seed")
    construction_seed = variant.get("seed") if isinstance(variant, dict) else execution_seed
    seed_policy = (
        "reset_after_variant" if isinstance(variant, dict) and type(execution_seed) is int
        else "construction_stream" if type(construction_seed) is int else "runtime"
    )
    call_items = calls if isinstance(calls, list) else []
    paths = sorted({
        path for call in call_items
        for path in call.get("paths", [])
        if isinstance(path, str)
    })
    anchors = [call.get("implementation") for call in call_items
               if isinstance(call.get("implementation"), dict)]
    anchors_available = sum(
        isinstance(anchor.get("source_sha256"), str) for anchor in anchors
    )
    anchor_digests = sorted({
        anchor["source_sha256"] for anchor in anchors
        if isinstance(anchor.get("source_sha256"), str)
    })
    call_evidence = []
    for call in call_items[:256]:
        implementation = call.get("implementation")
        call_evidence.append({
            "paths": [path for path in call.get("paths", []) if isinstance(path, str)],
            "module_type": call.get("module_type"),
            "invocation_index": call.get("invocation_index"),
            "implementation_source_sha256": implementation.get("source_sha256")
            if isinstance(implementation, dict) else None,
            "implementation_checkability": "source_digest_available"
            if isinstance(implementation, dict) and isinstance(implementation.get("source_sha256"), str)
            else "source_digest_unknown",
        })
    replayed = request.get("replay_input_from") is not None
    input_source_kind = (
        "snapshot_replay" if replayed
        else "dataset_sample" if isinstance(observation, dict)
        else "batch_factory" if isinstance(request.get("batch"), str) and request.get("batch")
        else "unknown"
    )
    model_anchor = implementation_anchor(model, request.get("project_root", "."))
    safe_model_anchor = {
        key: model_anchor.get(key)
        for key in ("qualified_type", "source_file", "source_sha256", "declaration_line")
        if key in model_anchor and not (
            key == "source_file" and isinstance(model_anchor.get(key), str)
            and Path(model_anchor["source_file"]).is_absolute()
        )
    }
    return {
        "schema": 1,
        "execution_status": execution_status,
        "input": {
            "source_kind": input_source_kind,
            "declared_batch_sha256": _declaration_digest(request.get("batch"))
            if request.get("batch") is not None else None,
            "dataset_fingerprint_sha256": _declaration_digest(observation.get("dataset_fingerprint"))
            if isinstance(observation, dict) and observation.get("dataset_fingerprint") is not None else None,
            "split_declared": isinstance(observation, dict) and observation.get("split") is not None,
            "stable_sample_ids_reported": sampling.get("stable_sample_ids") is True,
            "sample_count": len(sampling.get("sample_ids", []))
            if isinstance(sampling.get("sample_ids"), list) else None,
            "content": batch_content,
            "data_observation": _safe_data_observation(observation),
            "snapshot_status": snapshot_status or ("replayed" if replayed else "not_requested"),
            "checkability": "content_digest_available" if batch_content.get("status") == "available"
            else "content_unknown",
        },
        "output": {
            "structure_signature": output_signature,
            "status": "observed" if output_signature is not None and execution_status == "executed"
            else "partial" if execution_status == "failed"
            else "not_observed" if execution_status == "not_executed"
            else "historical_or_unknown" if execution_status == "reused"
            else "unknown",
            "values_fingerprinted": False,
            "scope": "bounded structure metadata only; no output tensor values retained",
        },
        "model": {
            "declared_model_sha256": _declaration_digest(request.get("model")),
            "implementation": safe_model_anchor,
            "variant_requested": variant is not None,
            "weights_before_forward": weight_report,
            "weights_after_forward": post_weight_report,
            "weight_comparison": (
                "matched" if isinstance(post_weight_report, dict)
                and weight_report.get("status") == "available"
                and post_weight_report.get("status") == "available"
                and hmac.compare_digest(weight_report["digest"], post_weight_report["digest"])
                else "changed" if isinstance(post_weight_report, dict)
                and weight_report.get("status") == "available"
                and post_weight_report.get("status") == "available"
                else "unknown_after_execution" if execution_status in {"executed", "failed"}
                else "not_observed"
            ),
        },
        "environment": environment_fingerprint(torch_module),
        "randomness": {
            "requested_seed_decimal": str(execution_seed) if type(execution_seed) is int else None,
            "construction_seed_decimal": str(construction_seed) if type(construction_seed) is int else None,
            "forward_rng_policy": seed_policy,
            "checkability": "seed_policy_reported_ambient_and_user_random_state_unknown",
        },
        "path_evidence": {
            "status": "observed" if execution_status == "executed" else
            "partial_failed" if execution_status == "failed" else
            "not_observed" if execution_status == "not_executed" else "historical_or_unobserved",
            "requested_path": request.get("path"),
            "requested_call": request.get("call", "all"),
            "observed_paths": paths,
            "observed_call_count": calls_observed if type(calls_observed) is int else len(call_items),
            "retained_call_count": len(call_items),
            "trace_truncated": bool(trace_truncated),
            "implementation_anchor_count": len(anchors),
            "implementation_anchor_available_count": anchors_available,
            "implementation_source_sha256": anchor_digests,
            "module_calls": call_evidence,
            "module_calls_truncated": len(call_items) > len(call_evidence),
            "scope": "registered module calls retained by this Probe; not all Python operations or unexecuted branches",
        },
    }


def _estimated_input_bytes(value, torch_module, seen=None, depth=0, item_count=None):
    """Conservative bound before hashing per-invocation values."""
    if seen is None:
        seen = set()
    if item_count is None:
        item_count = [0]
    item_count[0] += 1
    if item_count[0] > 4096:
        return None
    if depth > 32:
        return None
    kind = type(value)
    if value is None or kind in (bool, int, float):
        return 16
    if kind is str:
        return len(value.encode("utf-8"))
    if kind is torch_module.Tensor:
        try:
            return int(value.numel()) * int(value.element_size())
        except Exception:
            return None
    if kind in (tuple, list, dict):
        if id(value) in seen:
            return 1
        seen.add(id(value))
        total = 16
        items = value.items() if kind is dict else value
        for item in items:
            parts = item if kind is dict else (item,)
            for part in parts:
                amount = _estimated_input_bytes(
                    part, torch_module, seen, depth + 1, item_count
                )
                if amount is None:
                    return None
                total += amount
                if total > _MAX_BRANCH_FINGERPRINT_BYTES:
                    return total
        return total
    return None


def _signature_shape(value):
    """Keep structural metadata while excluding user scalar/tensor contents."""
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "tensor":
            return {
                "kind": "tensor",
                "shape": value.get("shape"),
                "dtype": value.get("dtype"),
                "layout": value.get("layout"),
                "memory_format": value.get("memory_format"),
                "device": value.get("device"),
                "requires_grad": value.get("requires_grad"),
            }
        if kind == "dict":
            items = value.get("items", {})
            return {
                "kind": "dict",
                "items": {
                    key: _signature_shape(item)
                    for key, item in sorted(items.items())
                    if not (key == "<truncated>" and value.get("items_truncated") is True)
                } if isinstance(items, dict) else {},
                "truncated": items.get("<truncated>") if isinstance(items, dict) and value.get("items_truncated") is True else 0,
                "keys_all_strings": value.get("keys_all_strings"),
                "observed_keys_all_strings": value.get("observed_keys_all_strings"),
            }
        if kind in {"tuple", "list"}:
            items = value.get("items", [])
            return {
                "kind": kind,
                "items": [_signature_shape(item) for item in items]
                if isinstance(items, list) else [],
                "truncated": value.get("truncated", 0),
            }
        if kind in {"opaque", "truncated"}:
            return {"kind": kind, "type": value.get("type")}
        return {"kind": "descriptor"}
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


def _input_signature(metadata):
    canonical = json.dumps(
        _signature_shape(metadata),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "structure-v3:" + hashlib.sha256(canonical).hexdigest()


def _port_paths(metadata, prefix, limit=128):
    paths = []

    def visit(value, path):
        if len(paths) >= limit:
            return
        if isinstance(value, dict) and value.get("kind") == "dict":
            items = value.get("items", {})
            if isinstance(items, dict) and items:
                # Metadata stringifies dictionary keys; do not turn ambiguous
                # non-string keys or the truncation marker into selectable ports.
                if value.get("observed_keys_all_strings", value.get("keys_all_strings")) is not True:
                    return
                for key, item in items.items():
                    if key == "<truncated>" and value.get("items_truncated") is True:
                        continue
                    key = str(key)
                    if len(key) > 128:
                        continue
                    suffix = ("." + key if key and not any(c in key for c in ".[]")
                              and not any(ord(c) < 32 or ord(c) == 127 for c in key)
                              else "[" + json.dumps(key, ensure_ascii=True) + "]")
                    visit(item, path + suffix)
                return
        elif isinstance(value, dict) and value.get("kind") in {"tuple", "list"}:
            items = value.get("items", [])
            if isinstance(items, list) and items:
                for index, item in enumerate(items):
                    visit(item, f"{path}[{index}]")
                return
        if len(path) <= 512:
            paths.append(path)

    visit(metadata, prefix)
    return paths


def _batch_identity(request):
    observation = request.get("_data_observation")
    sampling = observation.get("sampling", {}) if isinstance(observation, dict) else {}
    sample_ids = sampling.get("sample_ids") if isinstance(sampling, dict) else None
    stable = (
        isinstance(observation, dict)
        and isinstance(observation.get("dataset_fingerprint"), str)
        and bool(observation["dataset_fingerprint"])
        and isinstance(sample_ids, list)
        and bool(sample_ids)
        and sampling.get("stable_sample_ids") is True
    )
    if not stable:
        return "batch-" + uuid.uuid4().hex, False
    identity = {
        "source": observation.get("source"),
        "split": observation.get("split"),
        "dataset_fingerprint": observation["dataset_fingerprint"],
        "algorithm": sampling.get("algorithm"),
        "seed": sampling.get("seed"),
        "sample_ids": sample_ids,
    }
    canonical = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "batch-" + hashlib.sha256(canonical).hexdigest(), True


def implementation_anchor(module, project_root):
    """Identify readable Python module code without retaining source or values."""
    module_type = type(module)
    qualified_type = f"{module_type.__module__}.{module_type.__qualname__}"[:512]
    try:
        filename = inspect.getsourcefile(module_type)
        if not filename:
            return {"qualified_type": qualified_type, "source_file": None, "source_sha256": None}
        root = Path(project_root).resolve()
        source = Path(filename).resolve(strict=True)
        if not source.is_file() or source.stat().st_size > 8 * 1024 * 1024:
            raise OSError("source size or type unsupported")
        contents = source.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        environment = Path(sys.prefix).resolve()
        in_project_environment = environment != root and environment.is_relative_to(root) and source.is_relative_to(environment)
        runtime_roots = [Path(value).resolve() for name in ("purelib", "platlib", "stdlib", "platstdlib")
                         if (value := sysconfig.get_path(name))]
        in_runtime_library = any(source.is_relative_to(directory) for directory in runtime_roots)
        if not source.is_relative_to(root) or in_project_environment or in_runtime_library:
            if not qualified_type or "/" in qualified_type or "\\" in qualified_type:
                raise ValueError("external type identity unsupported")
            return {"qualified_type": qualified_type, "source_file": "@external:" + qualified_type,
                    "source_sha256": digest, "source_path": str(source)}
        relative = source.relative_to(root)
        # A declaration is only a navigation hint, not proof of the code
        # currently executing: modules can be redefined or changed after import.
        declaration_line = None
        parts = module_type.__qualname__.split(".")
        if parts and all(part.isidentifier() for part in parts):
            try:
                nodes = ast.parse(contents).body
                for part in parts:
                    matches = [node for node in nodes
                               if isinstance(node, ast.ClassDef) and node.name == part]
                    if len(matches) != 1:
                        break
                    declaration_line = matches[0].lineno
                    nodes = matches[0].body
                else:
                    parts = []
                if parts:
                    declaration_line = None
            except (SyntaxError, ValueError, UnicodeError, RecursionError, MemoryError):
                declaration_line = None
        return {
            "qualified_type": qualified_type,
            "source_file": relative.as_posix(),
            "source_sha256": digest,
            "declaration_line": declaration_line,
        }
    except (OSError, TypeError, ValueError, RuntimeError):
        return {"qualified_type": qualified_type, "source_file": None, "source_sha256": None}


def probe(
    model,
    request,
    *,
    torch_module,
    resolve_entrypoint,
    make_batch_fn,
    module_entries_fn,
    normalize_path_fn,
    resolve_module_fn,
    tree_fn,
    max_calls,
):
    replay_path = request.get("replay_input_from")
    capture_path = request.get("capture_input_to")
    if replay_path is not None:
        if not isinstance(replay_path, str) or capture_path is not None:
            raise UserError("INPUT_SNAPSHOT_INVALID", "输入快照请求无效。")
        try:
            payload = input_snapshot.read_file(replay_path)
            args, kwargs = input_snapshot.restore(payload, torch_module, expected_digest=request.get("expected_input_content"))
        except (OSError, input_snapshot.SnapshotError) as error:
            raise UserError("INPUT_SNAPSHOT_INVALID", "输入快照无法读取或无法通过内容校验。", "重新诊断并生成快照；没有执行候选前向。") from error
    else:
        args, kwargs, _ = make_batch_fn(request, torch_module, resolve_entrypoint)
    probe_id = "probe-" + uuid.uuid4().hex
    forward_id = "forward-" + uuid.uuid4().hex
    batch_id, batch_identity_stable = _batch_identity(request)
    model.train(bool(request.get("training", False)))
    input_tree = tree_fn(torch_module, {"args": args, "kwargs": kwargs})
    input_signature = _input_signature(input_tree)
    contract = request["_resolved_contract"] if "_resolved_contract" in request else load_contract(request)
    shape_inference = None
    current_source_inventory = {}
    batch_content = _fingerprint_forward_inputs(args, kwargs, torch_module)
    weights_report = weight_fingerprint_report(model, torch_module)
    # A plain batch factory cannot supply the stable dataset identity required
    # for cache reuse. With no contract, prepared provenance is never returned:
    # success/failure both collect a fresh post-forward report. Avoid an extra
    # environment inventory and source lookup in this common path.
    current_provenance = None
    if contract is not None or isinstance(request.get("_data_observation"), dict):
        current_provenance = _run_provenance(
            model, request, torch_module=torch_module, batch_content=batch_content,
            weight_report=weights_report, execution_status="prepared",
        )
    evidence_identity, cache_unavailable_reason = build_evidence_identity(
        model, request, torch_module=torch_module, input_content=batch_content, contract=contract,
        source_inventory_out=current_source_inventory, weights_report=weights_report,
    )
    previous_identity = load_latest_identity(request.get("project_root", ".")) if evidence_identity else None
    invalidated = invalidation_reasons(previous_identity, evidence_identity) if evidence_identity else []
    previous_data = previous_inventory = None
    if evidence_identity is not None and previous_identity is not None:
        previous_data, _, previous_inventory = load_evidence_with_inventory(
            request.get("project_root", "."), previous_identity
        )
    if evidence_identity is not None and not request.get("local_probe") and not request.get("_disable_evidence_cache") and request.get("expected_input_content") is None and capture_path is None and replay_path is None:
        cached_data, cache_key = load_evidence(request.get("project_root", "."), evidence_identity)
        if cached_data is not None:
            cached_data["historical_run_provenance"] = cached_data.get("run_provenance")
            current_provenance["execution_status"] = "reused"
            cached_data["run_provenance"] = current_provenance
            cached_data["input_sources"] = _input_source_compatibility(current_provenance)
            cached_data["evidence_cache"] = {"status": "reused", "key": cache_key, "invalidated": []}
            cached_data["forward_execution"] = "reused"
            # This describes the input prepared for this request, not a new
            # forward execution; it was checked against the cache identity.
            cached_data["batch_content"] = batch_content
            cached_data["coverage"] = (
                "复用与源码、权重、组件、契约、稳定样本身份及输入内容匹配的历史元数据证据；"
                "未执行本次前向，外部数据、环境变量及其他未记录状态仍可能改变结果。"
            )
            return cached_data
        cache_key = None
    elif cache_unavailable_reason:
        cache_key = None
    else:
        cache_key = None
    expected_content = request.get("expected_input_content")
    if expected_content is not None:
        if not isinstance(expected_content, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_content):
            raise UserError("INPUT_CONTENT_INVALID", "输入内容指纹格式无效。")
        if batch_content.get("status") != "available":
            raise UserError("INPUT_CONTENT_UNAVAILABLE", "当前前向输入无法完整核对内容。", "使用受支持的输入类型，或移除内容条件；未执行候选前向。", {"batch_content": batch_content, "forward_executed": False})
        if not hmac.compare_digest(expected_content, batch_content["digest"]):
            raise UserError("INPUT_CONTENT_MISMATCH", "当前前向输入内容与指定指纹不同。", "重新诊断或提供相同内容的批次；未执行候选前向。", {"batch_content": batch_content, "forward_executed": False})
    snapshot = {"status": "replayed", "format": input_snapshot.FORMAT} if replay_path is not None else None
    if capture_path is not None:
        if not isinstance(capture_path, str):
            raise UserError("INPUT_SNAPSHOT_INVALID", "输入快照路径无效。")
        try:
            payload = input_snapshot.encode(args, kwargs, torch_module)
        except Exception:
            snapshot = {"status": "unsupported", "format": input_snapshot.FORMAT}
        else:
            try:
                input_snapshot.write_new(capture_path, payload)
            except OSError as error:
                raise UserError("INPUT_SNAPSHOT_WRITE_FAILED", "无法创建输入快照。", "使用尚不存在的私有文件路径；没有执行前向。") from error
            snapshot = {"status": "captured", "format": input_snapshot.FORMAT}
    if contract is None and evidence_identity is not None:
        shape_inference = infer_with_fx_meta(model, args, kwargs, torch_module=torch_module)
    else:
        shape_inference = None
    if contract is not None:
        if contract.get("static", True) is True:
            shape_inference = infer_with_fx_meta(model, args, kwargs, torch_module=torch_module)
            if shape_inference["status"] == "failed":
                input_only_contract = {
                    "version": contract.get("version", 1),
                    "inputs": contract.get("inputs", {}),
                    "symbols": contract.get("symbols", {}),
                }
                contract_report = check_contract(input_only_contract, input_tree, {})
                contract_report["contract_content_sha256"] = request.get("_contract_content_sha256")
                contract_report["contract_identity_check"] = "matched" if request.get("if_contract_sha256") is not None else "not_requested"
                contract_report["status"] = "failed"
                contract_report["coverage"] = "静态维度错误已在真实前向前证实；输出端口尚未执行。"
                current_provenance["execution_status"] = "not_executed"
                raise UserError(
                    "MODEL_CONTRACT_FAILED",
                    "Meta/FX 推演发现静态维度不匹配，真实前向尚未执行。",
                    "根据 data.shape_inference.diagnostics 中的图节点、expected 和 actual 修正接线或 Layer 参数。",
                    {
                        "forward_execution": "not_executed",
                        "probe_id": probe_id,
                        "forward_id": forward_id,
                        "batch_id": batch_id,
                        "batch_identity_stable": batch_identity_stable,
                        "inputs": input_tree,
                        "input_signature": input_signature,
                        "shape_inference": shape_inference,
                        "contract_report": contract_report,
                        "run_provenance": current_provenance,
                        "input_sources": _input_source_compatibility(current_provenance),
                        "coverage": "静态推演证实的节点错误；动态分支和不支持算子未被证明。",
                    },
                )
        else:
            shape_inference = {"status": "not_requested", "coverage": "本次只检查声明契约与真实批次。"}
    modules, aliases = module_entries_fn(model)
    # Keep a separate bounded view of the selected subtree. A global trace
    # can fill up before a late layer executes in a large model.
    selected = normalize_path_fn(request.get("path"))
    selected_calls = []
    selected_observed = 0
    ids = {key: f"module-{index + 1}" for index, key in enumerate(modules)}
    counts = {key: 0 for key in modules}
    calls = []
    captured_inputs = {}
    implementation_anchors = {}
    branch_fingerprint_bytes = 0
    handles = []
    local = threading.local()
    observed = 0
    last_observed_call = None
    identity_trace = TensorIdentityTrace(torch_module)

    def before(module, call_args, call_kwargs):
        nonlocal observed, last_observed_call, branch_fingerprint_bytes, selected_observed
        key = id(module)
        counts[key] += 1
        observed += 1
        if not hasattr(local, "stack"):
            local.stack = []
        invocation_id = f"{forward_id}:call-{observed}"
        parent = local.stack[-1] if local.stack else None
        call_inputs = tree_fn(torch_module, {"args": call_args, "kwargs": call_kwargs})
        # Without a complete evidence identity no invocation can be reused,
        # so traversing every call's inputs merely to estimate hash cost is
        # unnecessary (and can be expensive for deeply nested structures).
        estimated_bytes = (
            _estimated_input_bytes((call_args, call_kwargs), torch_module)
            if evidence_identity is not None else None
        )
        if (
            evidence_identity is not None
            and estimated_bytes is not None
            and estimated_bytes <= _MAX_BRANCH_FINGERPRINT_BYTES - branch_fingerprint_bytes
        ):
            call_content_identity = _fingerprint_forward_inputs(
                call_args, call_kwargs, torch_module
            )
            if call_content_identity.get("status") == "available":
                branch_fingerprint_bytes += estimated_bytes
        else:
            call_content_identity = {
                "status": "unsupported",
                "reason": (
                    "complete evidence identity unavailable"
                    if evidence_identity is None
                    else "aggregate invocation fingerprint budget exceeded or input size unknown"
                ),
            }
        module_type = type(module)
        if module_type not in implementation_anchors:
            implementation_anchors[module_type] = implementation_anchor(
                module, request.get("project_root", ".")
            )
        last_observed_call = {
            "call_id": observed,
            "invocation_id": invocation_id,
            "probe_id": probe_id,
            "forward_id": forward_id,
            "batch_id": batch_id,
            "module_instance_id": ids[key],
            "paths": aliases[key],
            "alias_paths": aliases[key],
            "module_type": type(module).__name__,
            "implementation": implementation_anchors[module_type],
            "invocation_index": counts[key],
            "parent_call_id": parent[0] if parent else None,
            "parent_invocation_id": parent[2] if parent else None,
            "inputs": call_inputs,
            "input_signature": _input_signature(call_inputs),
            "input_content_identity": call_content_identity,
            "input_output_paths": _port_paths(call_inputs, "inputs"),
            "input_tensor_matches": identity_trace.inputs(call_args, call_kwargs),
        }
        selected_match = selected != "model" and any(
            path == selected or path.startswith(selected + ".") for path in aliases[key]
        )
        if selected_match:
            selected_observed += 1
        record = None
        in_global_trace = len(calls) < max_calls
        if in_global_trace:
            record = last_observed_call
            calls.append(record)
        if selected_match and len(selected_calls) < max_calls:
            record = last_observed_call
            selected_calls.append(record)
            if request.get("local_probe") and selected in aliases[key]:
                captured_inputs[invocation_id] = (call_args, call_kwargs)
        local.stack.append((observed, key, invocation_id, record, in_global_trace))

    def after(module, call_args, call_kwargs, output):
        if getattr(local, "stack", None):
            _, _, invocation_id, record, in_global_trace = local.stack.pop()
            target = record
            if (
                target is None
                and last_observed_call is not None
                and last_observed_call["invocation_id"] == invocation_id
            ):
                target = last_observed_call
            if target is not None:
                output_tree = tree_fn(torch_module, output)
                target["outputs"] = output_tree
                target["output_signature"] = _input_signature(output_tree)
                target["input_output_paths"].extend(_port_paths(output_tree, "outputs"))
                if in_global_trace:
                    identity_trace.outputs(output, invocation_id)

    for module in modules.values():
        handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))
    try:
        with torch_module.no_grad():
            output = model(*args, **kwargs)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        post_weights_report = weight_fingerprint_report(model, torch_module)
        origin = request.get("_model_origin")
        failure_origin = None
        if isinstance(origin, dict):
            params = origin.get("params")
            routes = params.get("input_routes") if isinstance(params, dict) else None
            bindings = origin.get("injection_evidence")
            failure_origin = {
                "scope": "partial_failure_context",
                "params": {"input_routes": routes} if isinstance(routes, dict) else {},
                "injection_evidence": [
                    {"slot": item.get("slot"), "registered_paths": item.get("registered_paths"),
                     "shared_with_slots": item.get("shared_with_slots")}
                    for item in bindings if isinstance(item, dict)
                ] if isinstance(bindings, list) else [],
            }
            if isinstance(origin.get("declared_routes"), dict):
                failure_origin["declared_routes"] = origin["declared_routes"]
        project_frames = []
        project_root = Path(request.get("project_root", ".")).resolve()
        for frame in traceback.extract_tb(error.__traceback__):
            try:
                frame_path = Path(frame.filename).resolve()
                relative = frame_path.relative_to(project_root)
            except (OSError, ValueError):
                continue
            frame_record = {
                "file": relative.as_posix(),
                "line": frame.lineno,
                "function": frame.name,
            }
            project_frames.append(frame_record)
        project_frames = project_frames[-16:]
        identities = {}
        for frame_record in project_frames:
            frame_path = project_root / frame_record["file"]
            if frame_path in identities:
                frame_record.update(identities[frame_path])
                continue
            identity = {}
            # This identifies the file as read after the exception. It does not
            # prove the running code was loaded from these exact bytes.
            try:
                before = frame_path.stat()
                if before.st_size <= 4 * 1024 * 1024 and frame_path.is_file():
                    with frame_path.open("rb") as source_file:
                        contents = source_file.read(4 * 1024 * 1024 + 1)
                    after = frame_path.stat()
                    if (len(contents) <= 4 * 1024 * 1024 and before.st_size == after.st_size
                            and before.st_mtime_ns == after.st_mtime_ns
                            and before.st_ctime_ns == after.st_ctime_ns
                            and before.st_ino == after.st_ino):
                        identity["source_sha256"] = hashlib.sha256(contents).hexdigest()
                        identity["identity_scope"] = "file-read-after-error"
            except OSError:
                pass
            identities[frame_path] = identity
            frame_record.update(identity)
        failed_provenance = _run_provenance(
            model, request, torch_module=torch_module, batch_content=batch_content,
            weight_report=weights_report, execution_status="failed",
            calls=calls, trace_truncated=observed > max_calls,
            calls_observed=observed,
            snapshot_status=snapshot.get("status") if snapshot is not None else "not_requested",
            post_weight_report=post_weights_report,
            output_signature=last_observed_call.get("output_signature")
            if isinstance(last_observed_call, dict) else None,
        )
        raise UserError("FORWARD_FAILED", f"真实前向失败：{type(error).__name__}: {error}",
                        "查看已采集调用和 --verbose 日志；没有执行反向或优化器更新。",
                        {
                            "forward_execution": "failed",
                            "probe_id": probe_id,
                            "forward_id": forward_id,
                            "batch_id": batch_id,
                            "batch_identity_stable": batch_identity_stable,
                            "batch_content": batch_content,
                            "run_provenance": failed_provenance,
                            "input_sources": _input_source_compatibility(failed_provenance),
                            **({"batch_snapshot": snapshot} if snapshot is not None else {}),
                            "input_signature": input_signature,
                            "calls": calls,
                            "calls_observed": observed,
                            "trace_truncated": observed > max_calls,
                            "tensor_identity_trace": identity_trace.coverage(),
                            "observed_mode": "train" if model.training else "eval",
                            **({"origin": failure_origin} if failure_origin is not None else {}),
                            "failure_location": {
                                "last_observed_module_call": last_observed_call,
                                "project_stack": project_frames[-16:],
                            },
                        "data_observation": _safe_data_observation(request.get("_data_observation")),
                        }) from error
    finally:
        for handle in handles:
            handle.remove()
    resolve_module_fn(model, selected)
    observed_calls = calls
    calls = observed_calls if selected == "model" else selected_calls
    chosen = request.get("call", "all")
    if chosen != "all":
        if not re.fullmatch(r"[1-9][0-9]*", str(chosen)):
            raise UserError("CALL_INVALID", "--call 需要正整数或 all。")
        # A specific call number refers to the selected module, not all descendants.
        calls = [call for call in calls if selected in call["paths"] and call["invocation_index"] == int(chosen)]
        if not calls:
            raise UserError("CALL_NOT_OBSERVED", "本次没有采集到选中的调用。", "调用序号只在本次执行内有效；先用 --call all 查看。")
    local_reports = []
    local_probe_truncated = False
    if request.get("local_probe"):
        if selected == "model":
            raise UserError("LOCAL_PROBE_PATH_REQUIRED", "局部 Probe 需要一个具体 Layer 路径。", "使用 --path model.encoder.block 或在 TTY 输入 Layer 路径。")
        local_module = resolve_module_fn(model, selected)
        eligible = [call for call in calls if selected in call["paths"]]
        local_probe_truncated = len(eligible) > 8 or selected_observed > max_calls
        for call in eligible[:8]:
            captured = captured_inputs.get(call["invocation_id"])
            if captured is None:
                continue
            local_reports.append(replay_invocation(
                local_module,
                captured[0],
                captured[1],
                {
                    "selected_path": selected,
                    "invocation_id": call["invocation_id"],
                    "invocation_index": call["invocation_index"],
                    "probe_id": probe_id,
                    "forward_id": forward_id,
                    "batch_id": batch_id,
                    "input_signature": call["input_signature"],
                },
                torch_module=torch_module,
                tree_fn=tree_fn,
            ))
        if not local_reports:
            local_reports.append({
                "status": "unobserved",
                "module_path": selected,
                "coverage": "本次调用未观察到该 Layer，或调用已超出保留上限；没有重放旧输入。",
            })
    output_tree = tree_fn(torch_module, output)
    output_signature = _input_signature(output_tree)
    post_weights_report = weight_fingerprint_report(model, torch_module)
    result = {
        "forward_execution": "executed",
        "probe_id": probe_id,
        "forward_id": forward_id,
        "batch_id": batch_id,
        "batch_identity_stable": batch_identity_stable,
        "batch_content": batch_content,
        **({"batch_snapshot": snapshot} if snapshot is not None else {}),
        "input_signature": input_signature,
        "output_signature": output_signature,
        "inputs": input_tree,
        "outputs": output_tree,
        "input_output_paths": _port_paths(input_tree, "inputs")
        + _port_paths(output_tree, "outputs"),
        "calls": calls,
        "evidence_calls": observed_calls,
        "calls_observed": observed,
        "selected_path": selected,
        "selected_calls_observed": observed if selected == "model" else selected_observed,
        "trace_truncated": observed > max_calls,
        "selected_trace_truncated": selected_observed > max_calls if selected != "model" else observed > max_calls,
        "tensor_identity_trace": identity_trace.coverage(),
        "observed_mode": "train" if model.training else "eval",
        "coverage": "仅本批次已采集的前向；未反向、未更新、不证明所有分支。",
    }
    result["run_provenance"] = _run_provenance(
        model, request, torch_module=torch_module, batch_content=batch_content,
        weight_report=weights_report, execution_status="executed",
        calls=observed_calls, trace_truncated=observed > max_calls,
        calls_observed=observed,
        snapshot_status=snapshot.get("status") if snapshot is not None else "not_requested",
        post_weight_report=post_weights_report,
        output_signature=output_signature,
    )
    result["input_sources"] = _input_source_compatibility(result["run_provenance"])
    if contract is not None:
        result["shape_inference"] = shape_inference
        result["contract_report"] = check_contract(contract, input_tree, output_tree, observed_calls)
        result["contract_report"]["contract_content_sha256"] = request.get("_contract_content_sha256")
        result["contract_report"]["contract_identity_check"] = "matched" if request.get("if_contract_sha256") is not None else "not_requested"
        if contract.get("modules") and observed > max_calls:
            result["contract_report"]["trace_truncated"] = True
            result["contract_report"]["coverage"] += " 模块调用记录已截断，契约未覆盖所有实际调用。"
            if result["contract_report"]["status"] == "observed-runtime":
                result["contract_report"]["status"] = "partial"
        if result["contract_report"]["status"] == "failed":
            raise UserError(
                "MODEL_CONTRACT_FAILED",
                "本次真实批次违反已声明的模型端口契约。",
                "根据 data.contract_report.checks 中的路径、expected 和 actual 修正契约或模型，再重新 probe。",
                result,
            )
    elif shape_inference is not None:
        result["shape_inference"] = shape_inference
    if evidence_identity is not None and not request.get("local_probe"):
        instance_calls = {}
        for call in observed_calls:
            instance_calls[call.get("module_instance_id")] = (
                instance_calls.get(call.get("module_instance_id"), 0) + 1
            )
        repeated = any(count > 1 for count in instance_calls.values())
        branch_report = reconcile_branch_evidence(
            previous_data,
            previous_inventory,
            previous_identity,
            evidence_identity,
            current_source_inventory,
            observed_calls,
            shape_inference=shape_inference,
            trace_truncated=observed > max_calls,
            repeated_module_instances=repeated,
        )
        result["evidence_cache"] = {
            "status": "not_stored",
            "reason": cache_unavailable_reason,
            "invalidated": invalidated,
            "branch_reuse": branch_report,
        }
    if request.get("local_probe"):
        result["local_probe_reports"] = local_reports
        result["local_probe_reports_truncated"] = local_probe_truncated
    if evidence_identity is not None and not request.get("local_probe") and not request.get("_disable_evidence_cache") and expected_content is None and capture_path is None and replay_path is None:
        cache_write_error = None
        if shape_inference is not None and shape_inference.get("status") == "inferred-static":
            if contract is None or result.get("contract_report", {}).get("status") == "observed-runtime":
                try:
                    cache_key = save_evidence(
                        request.get("project_root", "."), evidence_identity, result,
                        source_inventory=current_source_inventory,
                    )
                except OSError as error:
                    cache_key = None
                    cache_write_error = f"evidence cache write failed: {type(error).__name__}"
        result["evidence_cache"] = {
            "status": "stored" if cache_key else "not_stored",
            "key": cache_key,
            "invalidated": invalidated,
            "reason": None if cache_key else (
                cache_write_error or "contract has uncovered or unsupported ports"
                if contract is not None and result.get("contract_report", {}).get("status") != "observed-runtime"
                else cache_write_error or "static graph evidence is unavailable"
            ),
            "branch_reuse": branch_report if evidence_identity is not None else None,
        }
    else:
        result["evidence_cache"] = {
            "status": "not_stored",
            "reason": "local invocation replay is ephemeral" if request.get("local_probe") else cache_unavailable_reason,
            "invalidated": invalidated,
        }
    if request.get("_data_observation") is not None:
        result["data_observation"] = _safe_data_observation(request["_data_observation"])
    result.pop("evidence_calls", None)
    return result


def verify_gradients(model, request, *, torch_module, resolve_entrypoint, make_batch_fn, tree_fn):
    if not request.get("batch"):
        raise UserError("BATCH_REQUIRED", "梯度验证需要一个示例批次。", "使用 --batch module:example_batch。")
    loss_ref = request.get("loss")
    if not loss_ref:
        raise UserError("LOSS_REQUIRED", "梯度验证需要一个用户损失函数。", "使用 --loss module:loss_fn。")
    args, kwargs, batch = make_batch_fn(request, torch_module, resolve_entrypoint)
    loss_fn = resolve_entrypoint(loss_ref)
    model.train(bool(request.get("training", False)))
    model.zero_grad(set_to_none=True)
    try:
        outputs = model(*args, **kwargs)
        loss = loss_fn(outputs, batch)
        if not isinstance(loss, torch_module.Tensor) or loss.numel() != 1:
            raise UserError("LOSS_INVALID", "损失函数必须返回一个标量 Tensor。")
        if not loss.is_floating_point() or not loss.requires_grad:
            raise UserError("LOSS_INVALID", "损失必须是依赖模型计算图的浮点标量 Tensor。")
        if not bool(torch_module.isfinite(loss.detach()).all().item()):
            raise UserError("LOSS_NONFINITE", "损失值包含 NaN 或无穷大。")
        loss.backward()
    except UserError:
        raise
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        raise UserError(
            "BACKWARD_FAILED",
            f"前向、损失或反向失败：{type(error).__name__}: {error}",
            "检查损失函数输入、输出 shape 和可微路径；未执行优化器更新。",
        ) from error

    gradients = []
    disconnected = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is None:
            disconnected.append(name)
            continue
        values = grad.coalesce().values() if grad.is_sparse else grad
        norm = float(values.detach().norm().item())
        finite = bool(torch_module.isfinite(values).all().item()) and math.isfinite(norm)
        gradients.append({"name": name, "finite": finite, "norm": norm})
        if not finite:
            nonfinite.append(name)
    if not gradients:
        raise UserError(
            "GRADIENT_MISSING",
            "损失没有为任何可训练参数产生梯度。",
            "检查损失是否依赖模型输出，以及参数是否被冻结。",
        )
    if nonfinite:
        raise UserError(
            "GRADIENT_NONFINITE",
            f"以下参数的梯度含 NaN 或无穷大：{', '.join(nonfinite)}",
            "尚未执行参数更新。检查数值范围、损失缩放和精度设置。",
        )
    result = {
        "inputs": tree_fn(torch_module, {"args": args, "kwargs": kwargs}),
        "outputs": tree_fn(torch_module, outputs),
        "loss": float(loss.detach().item()),
        "gradients": gradients,
        "disconnected_parameters": disconnected,
        "gradient_count": len(gradients),
        "observed_mode": "train" if model.training else "eval",
        "coverage": "已执行指定批次的一次真实前向、用户损失和反向；未更新参数，不代表完整训练循环。",
    }
    if request.get("_data_observation") is not None:
        result["data_observation"] = _safe_data_observation(request["_data_observation"])
    return result


