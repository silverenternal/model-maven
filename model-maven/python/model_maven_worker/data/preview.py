"""Request-scoped preview generation and completion manifests."""

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..protocol import UserError
from ..runtime.entrypoints import configure_project_import_roots
from .batch import make_batch
from .renderers import RENDERER_VERSION, _structure_summary, invoke_custom_renderer, render_auto, render_summary
from .transforms import TRANSFORM_REGISTRY_VERSION, run_transform_pipeline


PREVIEW_PROTOCOL_VERSION = 1
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def preview_signature(identity, renderer="builtin.auto", renderer_version=RENDERER_VERSION):
    """Hash declared preview inputs without persisting sample values or credentials."""
    if not isinstance(identity, dict) or not identity:
        raise UserError("PREVIEW_IDENTITY_INVALID", "预览 identity 必须是非空对象。")
    payload = {
        "protocol_version": PREVIEW_PROTOCOL_VERSION,
        "renderer": renderer,
        "renderer_version": str(renderer_version),
        "identity": identity,
    }
    try:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise UserError("PREVIEW_IDENTITY_INVALID", f"预览 identity 不是有限 JSON 数据：{error}") from error
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_preview(request, samples, output_directory, custom_renderer=None):
    """Render one immutable request directory and atomically write its completion manifest."""
    if not isinstance(request, dict):
        raise UserError("PREVIEW_REQUEST_INVALID", "预览请求必须是 JSON 对象。")
    request_id = request.get("request_id")
    generation = request.get("generation")
    renderer = request.get("renderer", "builtin.auto")
    identity = request.get("identity")
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise UserError("PREVIEW_REQUEST_INVALID", "request_id 必须是 1 到 96 位安全文件名字符。")
    if type(generation) is not int or generation < 1:
        raise UserError("PREVIEW_REQUEST_INVALID", "generation 必须是正整数。")
    if not isinstance(samples, (list, tuple)) or not samples:
        raise UserError("PREVIEW_SAMPLES_EMPTY", "预览需要至少一个真实样本。", "先确认数据切分与采样预算，再生成预览。")
    if renderer not in {"builtin.auto", "builtin.summary", "project.custom"}:
        raise UserError("PREVIEW_RENDERER_UNSUPPORTED", f"不支持的 renderer：{renderer!r}。")
    version = request.get("renderer_version", RENDERER_VERSION if renderer.startswith("builtin.") else None)
    if version is None:
        raise UserError("PREVIEW_RENDERER_VERSION_REQUIRED", "自定义 renderer 必须声明 renderer_version 并纳入预览身份。")
    signature = preview_signature(identity, renderer, version)
    supplied_signature = request.get("input_signature")
    if supplied_signature is not None and supplied_signature != signature:
        raise UserError(
            "PREVIEW_SIGNATURE_MISMATCH",
            "请求 input_signature 与声明的预览输入不匹配。",
            "重新从当前数据、样本、种子、变换、代码/权重和 renderer 版本计算请求签名。",
            {"expected": signature, "actual": supplied_signature},
        )

    directory = Path(output_directory).expanduser()
    if directory.is_symlink():
        raise UserError("PREVIEW_PATH_INVALID", "预览请求目录不能是符号链接。")
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    title = request.get("title", "Model-Maven data preview")
    if not isinstance(title, str) or not title.strip() or len(title) > 160:
        raise UserError("PREVIEW_REQUEST_INVALID", "title 必须是 1 到 160 个字符。")
    try:
        if renderer == "builtin.auto":
            rendered = render_auto(samples, directory, title)
        elif renderer == "builtin.summary":
            rendered = render_summary(samples, directory, title)
        else:
            if not callable(custom_renderer):
                raise UserError("PREVIEW_RENDERER_UNAVAILABLE", "请求指定了项目 renderer，但没有提供可调用入口。")
            rendered = invoke_custom_renderer(custom_renderer, samples, directory, request)
    except UserError:
        raise
    except Exception as error:
        raise UserError(
            "PREVIEW_RENDER_FAILED",
            f"预览生成失败：{type(error).__name__}: {error}",
            "原成功预览应继续显示并标记过期；修复 renderer 或输入后重新生成。",
            {"request_id": request_id, "generation": generation},
        ) from error

    artifacts = _verify_and_hash_artifacts(directory, rendered.get("artifacts"))
    manifest = {
        "schema_version": PREVIEW_PROTOCOL_VERSION,
        "request_id": request_id,
        "generation": generation,
        "base_signature": request.get("base_signature"),
        "input_signature": signature,
        "renderer": rendered["renderer"],
        "renderer_version": rendered["renderer_version"],
        "scope": _safe_json(request.get("scope", {}), "scope"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": rendered["sample_count"],
        "rendered_image_count": rendered["rendered_image_count"],
        "truncated": rendered["truncated"],
        "artifacts": artifacts,
    }
    _atomic_json(directory / "complete.json", manifest)
    return manifest


def run_preview_operation(request, torch_module, resolve_entrypoint, source_manifest):
    """Sample a project source, apply its declared transforms, and render one request."""
    project_root = Path(request.get("project_root", "")).expanduser().resolve()
    if not project_root.is_dir():
        raise UserError("PROJECT_UNAVAILABLE", "预览项目目录不存在。")
    if str(project_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(project_root))
    configure_project_import_roots(request)
    preview_request = request.get("preview_request")
    if not isinstance(preview_request, dict):
        raise UserError("PREVIEW_REQUEST_INVALID", "worker 请求缺少 preview_request 对象。")
    source = request.get("preview_source")
    if not isinstance(source, dict) or source.get("kind") not in {"dataset", "batch"}:
        raise UserError("PREVIEW_SOURCE_INVALID", "预览来源必须是 dataset 或 batch 工厂。")
    source_entrypoint = source.get("entrypoint")
    parameters = source.get("parameters", {})
    binding = source.get("binding")
    if binding is not None and (source["kind"] != "batch" or binding != "single-raw"):
        raise UserError("PREVIEW_BINDING_UNSUPPORTED", "数据预览只支持批次工厂的 single-raw 整体模式，或不指定绑定。",
                        "选择批次工厂的整体预览，或移除预览绑定；尚未调用数据工厂。")
    if not isinstance(source_entrypoint, str) or not source_entrypoint.strip() or not isinstance(parameters, dict):
        raise UserError("PREVIEW_SOURCE_INVALID", "预览来源需要 module:callable 和 JSON 对象参数。")

    batch_request = {
        "preview_only": True,
        "batch": source_entrypoint if source["kind"] == "batch" else None,
        "batch_params": parameters if source["kind"] == "batch" else {},
        "batch_binding": source.get("binding") if source["kind"] == "batch" else None,
        "batch_fingerprint": None,
        "dataset": {
            "entrypoint": source_entrypoint,
            "parameters": parameters,
            "split": request.get("split"),
            "samples": request.get("samples", 12),
            "seed": request.get("seed", 0),
            "fingerprint": request.get("fingerprint"),
            "scan_limit": request.get("scan_limit"),
        } if source["kind"] == "dataset" else None,
        "data_samples": request.get("samples", 12),
        "data_split": request.get("split"),
        "data_fingerprint": request.get("fingerprint"),
        "seed": request.get("seed", 0),
    }
    _, _, sampled_value = make_batch(batch_request, torch_module, resolve_entrypoint)
    observation = batch_request.get("_data_observation")
    count = observation.get("sampling", {}).get("actual_samples") if isinstance(observation, dict) else None
    # This built-in factory explicitly returns a named BCHW image batch and a
    # corresponding named target. Arbitrary batch factories remain unsplit.
    graph_image = source["kind"] == "batch" and source_entrypoint == "model_maven_worker.data.presets:graph_image_batch"
    effective_split = request.get("split")
    if graph_image:
        input_name, target_name = parameters.get("input_name"), parameters.get("target_name")
        kwargs = sampled_value.get("kwargs") if isinstance(sampled_value, dict) else None
        loss_inputs = sampled_value.get("loss_inputs") if isinstance(sampled_value, dict) else None
        targets = loss_inputs.get("targets") if isinstance(loss_inputs, dict) else None
        dataset = loss_inputs.get("dataset") if isinstance(loss_inputs, dict) else None
        image = kwargs.get(input_name) if isinstance(kwargs, dict) and isinstance(input_name, str) else None
        target = targets.get(target_name) if isinstance(targets, dict) and isinstance(target_name, str) else None
        if (not isinstance(image, torch_module.Tensor) or image.ndim != 4 or not isinstance(target, torch_module.Tensor)
                or target.ndim < 1 or image.shape[0] < 1 or image.shape[0] != target.shape[0]
                or not isinstance(dataset, dict) or dataset.get("split") not in {"train", "validation", "test"}):
            raise UserError("GRAPH_IMAGE_PREVIEW_INVALID", "图像批次的输入和目标不符合已声明的批次契约。")
        count = int(image.shape[0])
        effective_split = dataset["split"]
    if type(count) is int and count > 0:
        samples = _split_preview_batch(sampled_value, count)
        sample_interpretation = "graph_image_batch_samples" if graph_image else "dataset_collated_samples"
    else:
        # Arbitrary factories can return a single matrix, an already batched
        # tensor, or nested inputs. Shape alone cannot distinguish them.
        samples = [sampled_value]
        sample_interpretation = "whole_factory_output_batch_axis_unknown"
    samples = samples[:request.get("samples", 12)]
    if not samples:
        raise UserError("PREVIEW_SAMPLES_EMPTY", "来源没有返回可预览样本。")

    transforms = request.get("transforms", [])
    if not isinstance(transforms, list):
        raise UserError("TRANSFORM_PIPELINE_INVALID", "transforms 必须是 JSON 数组。")
    seed = request.get("seed", 0)
    if type(seed) is not int:
        raise UserError("TRANSFORM_SEED_INVALID", "变换种子必须是整数。")
    transform_observations = []
    if transforms:
        transformed = []
        for index, sample in enumerate(samples):
            result = run_transform_pipeline(
                transforms,
                sample,
                resolve_entrypoint=resolve_entrypoint,
                seed=seed + index,
            )
            transformed.append(result["output"])
            transform_observations.append({
                "sample_index": index,
                "steps": result["steps"],
                "output_port": result["output_port"],
            })
        samples = transformed

    modules = source_manifest(project_root)
    source_digest = hashlib.sha256(
        json.dumps(modules, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    data_observation = observation or {
        "source": source_entrypoint,
        "split": effective_split,
        "augment_selection": request.get("augment_selection"),
        "fingerprint": request.get("fingerprint"),
        "sample_count": len(samples),
        "stable_sample_ids": False,
        "sample_ids": [],
    }
    sample_ids = data_observation.get("sampling", {}).get("sample_ids", [])
    stable_sample_ids = bool(data_observation.get("sampling", {}).get("stable_sample_ids"))
    if source["kind"] == "batch":
        # A declared data version identifies neither the factory's internal
        # state nor a particular returned sample at an index.
        stable_sample_ids = False
        sample_ids = []
    renderer = preview_request.get("renderer", "builtin.auto")
    renderer_version = preview_request.get("renderer_version", RENDERER_VERSION)
    custom_renderer = None
    if renderer == "project.custom":
        renderer_entrypoint = preview_request.get("renderer_entrypoint")
        if not isinstance(renderer_entrypoint, str) or not renderer_entrypoint.strip():
            raise UserError("PREVIEW_RENDERER_INVALID", "项目 renderer 缺少 renderer_entrypoint。")
        custom_renderer = resolve_entrypoint(renderer_entrypoint)
    identity = {
        "source_kind": source["kind"],
        "source_entrypoint": source_entrypoint,
        "batch_binding": source.get("binding") if source["kind"] == "batch" else None,
        "source_parameters_sha256": hashlib.sha256(
            json.dumps(parameters, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "source_manifest_sha256": source_digest,
        "data_fingerprint": request.get("fingerprint"),
        "split": effective_split,
        "sampling_algorithm": data_observation.get("sampling", {}).get("algorithm", "batch-factory-v1"),
        "sample_ids": sample_ids,
        "stable_sample_ids": stable_sample_ids,
        "unstable_request_id": None if stable_sample_ids else preview_request["request_id"],
        "seed": seed,
        "transforms_sha256": hashlib.sha256(
            json.dumps(transforms, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "transform_registry_version": TRANSFORM_REGISTRY_VERSION,
        "runtime_paths": {"working_directory": request.get("runtime", {}).get("working_directory"),
                          "project_import_roots": request.get("runtime", {}).get("project_import_roots")},
    }
    signature = preview_signature(identity, renderer, renderer_version)
    scope = {
        "source_kind": source["kind"],
        "source_entrypoint": source_entrypoint,
        "runtime_paths": identity["runtime_paths"],
        "batch_binding": source.get("binding") if source["kind"] == "batch" else None,
        "split": effective_split,
        "preview_split_request": request.get("split"),
        "augment_selection": request.get("augment_selection"),
        "requested_samples": request.get("samples", 12),
        "actual_samples": len(samples),
        "sample_interpretation": sample_interpretation,
        "seed": seed,
        "seed_decimal": str(seed),
        "sampling_algorithm": identity["sampling_algorithm"],
        "stable_sample_ids": stable_sample_ids,
        "sample_ids_sha256": hashlib.sha256(
            json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if sample_ids else None,
        "data_observation": {
            key: observation.get(key)
            for key in ("adapter", "adapter_version", "coverage", "capabilities")
        } if isinstance(observation, dict) else None,
    }
    manifest = render_preview(
        {
            **preview_request,
            "identity": identity,
            "input_signature": signature,
            "base_signature": preview_request.get("base_signature"),
            "renderer_version": renderer_version,
            "scope": scope,
        },
        samples,
        preview_request.get("request_directory"),
        custom_renderer=custom_renderer,
    )
    return {
        **manifest,
        "data_observation": observation,
        "stable_sample_ids": stable_sample_ids,
        "transform_observations": transform_observations,
        # Bounded, value-free structure for the terminal report. HTML remains
        # the full artifact and may also include labels or sample identifiers.
        "structure_summaries": [_structure_summary(sample) for sample in samples[:3]]
        if renderer == "builtin.summary" else [],
    }


def _split_preview_batch(value, count):
    if hasattr(value, "shape") and hasattr(value, "device"):
        shape = tuple(int(item) for item in value.shape)
        # This path is only used after Dataset sampling reports a count and
        # the adapter has collated the sampled values, not for arbitrary
        # factory outputs with unknown batch semantics.
        if shape and shape[0] >= count:
            return [value[index] for index in range(count)]
        return [value]

    def at(items, index):
        # Non-batched metadata is shared across the observed samples. It must
        # not prevent batched tensors in another field from being split.
        return items[index] if len(items) == count else items[0]

    if isinstance(value, dict):
        children = {
            key: [item] if key in {"metadata", "dataset"} and isinstance(item, dict)
            else _split_preview_batch(item, count)
            for key, item in value.items()
        }
        if (count == 1 or any(len(items) == count for items in children.values())) and all(
            len(items) in {1, count} for items in children.values()
        ):
            return [{key: at(items, index) for key, items in children.items()} for index in range(count)]
        return [value]
    if isinstance(value, tuple):
        children = [_split_preview_batch(item, count) for item in value]
        if (count == 1 or any(len(items) == count for items in children)) and all(
            len(items) in {1, count} for items in children
        ):
            return [tuple(at(items, index) for items in children) for index in range(count)]
        return [value]
    if isinstance(value, list):
        children = [_split_preview_batch(item, count) for item in value]
        if any(len(items) == count and count > 1 for items in children):
            if all(len(items) in {1, count} for items in children):
                return [[at(items, index) for items in children] for index in range(count)]
            return [value]
        if len(value) == count and count > 1:
            return value
        if count == 1 and all(len(items) == 1 for items in children):
            return [[items[0] for items in children]]
        return [value]
    return [value]


def _verify_and_hash_artifacts(directory, artifacts):
    if not isinstance(artifacts, list) or not artifacts:
        raise UserError("PREVIEW_ARTIFACTS_MISSING", "renderer 没有生成可用预览产物。")
    verified = []
    seen = set()
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise UserError("PREVIEW_ARTIFACT_INVALID", f"artifacts[{index}] 必须是对象。")
        path_value = item.get("path")
        kind = item.get("kind")
        media_type = item.get("media_type")
        if (
            not isinstance(path_value, str)
            or not isinstance(kind, str)
            or not kind
            or not isinstance(media_type, str)
            or not media_type
        ):
            raise UserError("PREVIEW_ARTIFACT_INVALID", f"artifacts[{index}] 需要 path、kind 和 media_type。")
        relative = Path(path_value)
        if relative.is_absolute() or not relative.parts or any(part in {".", ".."} for part in relative.parts):
            raise UserError("PREVIEW_ARTIFACT_INVALID", f"artifacts[{index}].path 必须是请求目录内的相对路径。")
        candidate = directory / relative
        current = directory
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise UserError("PREVIEW_ARTIFACT_INVALID", f"artifacts[{index}] 路径不能经过符号链接。")
        if relative.as_posix() == "complete.json":
            raise UserError("PREVIEW_ARTIFACT_INVALID", "complete.json 是保留的完成标记路径。")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(directory)
        except ValueError as error:
            raise UserError("PREVIEW_ARTIFACT_INVALID", f"artifacts[{index}] 越出了请求目录。") from error
        if resolved.is_symlink() or not resolved.is_file():
            raise UserError("PREVIEW_ARTIFACT_MISSING", f"预览产物不存在、不是普通文件或为符号链接：{path_value}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise UserError("PREVIEW_ARTIFACT_INVALID", f"renderer 重复返回产物路径：{normalized}")
        seen.add(normalized)
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        verified.append({
            "kind": kind,
            "path": normalized,
            "media_type": media_type,
            "size_bytes": resolved.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return verified


def _safe_json(value, field):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise UserError("PREVIEW_REQUEST_INVALID", f"{field} 必须是有限 JSON 数据：{error}") from error


def _atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
