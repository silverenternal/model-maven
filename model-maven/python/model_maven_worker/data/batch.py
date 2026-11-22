"""Resolve a user batch factory and bind its result to a model call."""

from ..protocol import UserError


def make_batch(request, torch_module, resolve_entrypoint):
    source = request.get("batch")
    dataset_request = request.get("dataset")
    if source and dataset_request:
        raise UserError("DATA_SOURCE_AMBIGUOUS", "一次 probe 只能选择 batch 工厂或 Dataset 工厂。")
    if dataset_request:
        batch = _dataset_batch(request, dataset_request, torch_module, resolve_entrypoint)
    elif not source:
        raise UserError("BATCH_REQUIRED", "前向只需要再提供一个示例批次，无需训练配置。", "使用 --batch package.module:example_batch。")
    else:
        try:
            batch = resolve_entrypoint(source)(**request.get("batch_params", {}))
        except UserError:
            raise
        except Exception as error:
            raise UserError("BATCH_CONSTRUCTION_FAILED", str(error)) from error
    binding = request.get("batch_binding")
    if not dataset_request and _is_dataset(batch, torch_module):
        sampled = _sample_dataset_instance(
            source,
            batch,
            request,
            torch_module,
            parameters=request.get("batch_params", {}),
        )
        batch = sampled
        binding = request.get("batch_binding")
    if request.get("preview_only") is True:
        return (), {}, batch
    envelope_keys = {"args", "kwargs", "loss_inputs", "metadata"}
    is_envelope = (
        isinstance(batch, dict)
        and {"args", "kwargs"}.issubset(batch)
        and set(batch).issubset(envelope_keys)
    )
    if binding == "envelope" or (binding is None and is_envelope):
        if not isinstance(batch, dict) or not isinstance(batch.get("args"), (list, tuple)) or not isinstance(batch.get("kwargs"), dict):
            raise UserError("BATCH_BINDING_INVALID", "envelope 需要 {args: [...], kwargs: {...}}。")
        return tuple(batch["args"]), batch["kwargs"], batch
    if binding == "single" or (binding is None and isinstance(batch, torch_module.Tensor)):
        return (batch,), {}, batch
    if binding == "args" and isinstance(batch, (tuple, list)):
        return tuple(batch), {}, batch
    if binding == "kwargs" and isinstance(batch, dict):
        return (), batch, batch
    raise UserError("BATCH_BINDING_REQUIRED", "无法无歧义地将批次绑定到 forward。", "显式选择 --batch-binding single|args|kwargs，或返回 args/kwargs envelope。")


def _dataset_batch(request, specification, torch_module, resolve_entrypoint):
    DatasetAdapter = _dataset_adapter()
    if not isinstance(specification, dict):
        raise UserError("DATASET_SPEC_INVALID", "dataset 必须是包含 entrypoint 的对象。")
    allowed = {"entrypoint", "parameters", "split", "samples", "seed", "fingerprint", "scan_limit"}
    unexpected = sorted(set(specification) - allowed)
    if unexpected:
        raise UserError("DATASET_SPEC_INVALID", f"dataset 包含未知字段：{', '.join(unexpected)}")
    source = specification.get("entrypoint")
    parameters = specification.get("parameters", {})
    if not isinstance(source, str) or not source.strip():
        raise UserError("DATASET_SPEC_INVALID", "dataset.entrypoint 必须是 module:callable。")
    if not isinstance(parameters, dict):
        raise UserError("DATASET_SPEC_INVALID", "dataset.parameters 必须是对象。")
    try:
        factory = resolve_entrypoint(source)
        adapter = DatasetAdapter(source, factory, parameters)
        count = specification.get("samples", 1)
        seed = specification.get("seed")
        if seed is None:
            seed = request.get("seed")
        if seed is None:
            seed = 0
        collate = _dataset_collate(torch_module)
        sampled = adapter.sample(
            count=count,
            seed=seed,
            split=specification.get("split"),
            fingerprint=specification.get("fingerprint"),
            collate=collate,
            scan_limit=specification.get("scan_limit"),
        )
    except UserError:
        raise
    except Exception as error:
        raise UserError("DATASET_CONSTRUCTION_FAILED", str(error)) from error
    _save_observation(request, sampled)
    return sampled.batch


def _sample_dataset_instance(source, dataset, request, torch_module, parameters=None):
    DatasetAdapter = _dataset_adapter()
    adapter = DatasetAdapter(source or "batch-factory", lambda **_: dataset, parameters)
    sampled = adapter.sample_instance(
        dataset,
        count=request.get("data_samples", 1),
        seed=request.get("seed") if request.get("seed") is not None else 0,
        split=request.get("data_split"),
        fingerprint=request.get("data_fingerprint"),
        collate=_dataset_collate(torch_module),
    )
    _save_observation(request, sampled)
    return sampled.batch


def _save_observation(request, sampled):
    request["_data_observation"] = sampled.observation.to_dict()
    request["_data_state"] = sampled.state


def _is_dataset(value, torch_module):
    if isinstance(value, (dict, list, tuple, str, bytes)):
        return False
    if torch_module is not None:
        tensor_type = getattr(torch_module, "Tensor", None)
        if isinstance(tensor_type, type) and isinstance(value, tensor_type):
            return False
    return (hasattr(value, "__len__") and hasattr(value, "__getitem__")) or hasattr(value, "__iter__")


def _dataset_collate(torch_module):
    """Use PyTorch's native collator for Dataset items when available."""
    if torch_module is None:
        return None
    default_collate = getattr(getattr(torch_module, "utils", None), "data", None)
    default_collate = getattr(default_collate, "default_collate", None)
    return default_collate


def _dataset_adapter():
    try:
        from .adapters import DatasetAdapter
    except ImportError as error:
        raise UserError(
            "DATA_ADAPTER_UNAVAILABLE",
            "当前 worker 没有装载 Dataset 适配器模块。",
            "更新到包含完整 worker 资产清单的 Model-Maven 构建后重试。",
        ) from error
    return DatasetAdapter

