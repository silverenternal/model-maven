"""Small classification presets that use Model-Maven's native data/train contracts.

Preset factories are conveniences, not a separate training engine. They return
ordinary ``torch.nn.Module`` objects and ordinary batch/step callables, so the
same inspect, try, train, checkpoint, and evaluation paths remain available.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

from ..protocol import UserError
from .classification import load_project_exclusions, scan_class_folder
from .sampling import split_classification_index
from .transforms import run_transform_pipeline


PRESET_VERSION = 1
RESNET18_WEIGHT_ID = "imagenet1k-v1"
RESNET18_WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET18_WEIGHT_FILENAME = "resnet18-f37072fd.pth"
RESNET18_WEIGHT_SHA256_PREFIX = "f37072fd"
RESNET18_WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
_CLASSIFICATION_BATCH_CURSORS = {}
_GRAPH_IMAGE_EVIDENCE = {}
_GRAPH_IMAGE_CURSORS = {}
_GRAPH_IMAGE_INDEX_CACHES = OrderedDict()


def describe_presets():
    """Describe built-ins without importing torch, torchvision, or Pillow."""
    return {
        "version": PRESET_VERSION,
        "presets": {
            "tiny-cnn": {
                "task": "image-classification",
                "model": "model_maven_worker.data.presets:build_tiny_cnn",
                "weights": ["random"],
                "default_input": {"channels": 3, "height": 32, "width": 32},
                "data": "local class-folder",
                "training_step": "model_maven_worker.data.presets:classification_step",
            },
            "resnet18": {
                "task": "image-classification",
                "model": "model_maven_worker.data.presets:build_resnet18",
                "weights": ["random", RESNET18_WEIGHT_ID],
                "default_input": {"channels": 3, "height": 224, "width": 224},
                "data": "local class-folder",
                "training_step": "model_maven_worker.data.presets:classification_step",
                "weight_source": RESNET18_WEIGHT_URL,
                "weight_sha256": RESNET18_WEIGHT_SHA256,
                "download_is_explicit": True,
            },
        },
    }


def build_tiny_cnn(num_classes, in_channels=3, width=16):
    """Build a compact native PyTorch classifier with spatially flexible inputs."""
    torch = _torch()
    _validate_model_dimensions(num_classes, in_channels, width)
    nn = torch.nn
    model = nn.Sequential(
        nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(width * 2, num_classes),
    )
    model.model_maven_weight_plan = {"plan": "random", "source": None}
    return model


def resnet18_weights_status(cache_dir=None):
    """Report whether the pinned official ResNet18 checkpoint is cached."""
    directory = _torch_hub_cache_dir(cache_dir)
    checkpoint = directory / RESNET18_WEIGHT_FILENAME
    exists = checkpoint.is_file()
    evidence = {
        "weight_id": RESNET18_WEIGHT_ID,
        "source": RESNET18_WEIGHT_URL,
        "cache_path": str(checkpoint),
        "cached": exists,
        "download_required": not exists,
        "expected_sha256": RESNET18_WEIGHT_SHA256,
    }
    if exists:
        evidence["sha256"] = _sha256(checkpoint)
        evidence["sha256_verified"] = evidence["sha256"] == RESNET18_WEIGHT_SHA256
        if not evidence["sha256_verified"]:
            evidence["cached"] = False
            evidence["download_required"] = True
            evidence["cache_error"] = "sha256_mismatch"
    return evidence


def build_resnet18(
    num_classes,
    weights="random",
    allow_download=False,
    cache_dir=None,
):
    """Build torchvision ResNet18; weight fetching requires explicit opt-in.

    ``weights`` accepts ``random`` or the pinned ``imagenet1k-v1`` checkpoint.
    A pretrained checkpoint is read from the torch hub cache by default. This
    function never contacts the network unless ``allow_download=True``.
    """
    torch = _torch()
    _validate_model_dimensions(num_classes, 3, 1)
    if weights not in {"random", RESNET18_WEIGHT_ID}:
        raise UserError(
            "PRESET_WEIGHTS_UNSUPPORTED",
            f"不支持 ResNet18 权重计划 {weights!r}。",
            f"可选值：random、{RESNET18_WEIGHT_ID}。",
            {"available": ["random", RESNET18_WEIGHT_ID]},
        )
    checkpoint = None
    weight_evidence = {"plan": weights, "source": None, "cache_path": None, "sha256": None}
    if weights == RESNET18_WEIGHT_ID:
        status = resnet18_weights_status(cache_dir)
        weight_evidence.update(status)
        if not status["cached"]:
            if not allow_download:
                raise UserError(
                    "PRETRAINED_WEIGHTS_NOT_CACHED",
                    "ResNet18 预训练权重尚未缓存；当前没有下载。",
                    "先明确执行带 --download 的权重准备操作，或改用 weights=random。",
                    {**status, "download_flag_required": True},
                )
            try:
                checkpoint = torch.hub.load_state_dict_from_url(
                    RESNET18_WEIGHT_URL,
                    model_dir=str(Path(cache_dir).expanduser()) if cache_dir else None,
                    map_location="cpu",
                    progress=True,
                    check_hash=True,
                )
            except Exception as error:
                raise UserError(
                    "PRETRAINED_WEIGHTS_DOWNLOAD_FAILED",
                    f"无法准备 ResNet18 权重：{type(error).__name__}: {error}",
                    "检查网络/代理或离线准备权重；模型和数据项目没有被修改。",
                    {**status, "source": RESNET18_WEIGHT_URL},
                ) from error
            status = resnet18_weights_status(cache_dir)
            if not status["cached"]:
                raise UserError(
                    "PRETRAINED_WEIGHTS_CACHE_INVALID",
                    "下载完成后权重缓存未通过来源哈希检查。",
                    "删除该缓存文件后重试，或选择随机初始化。",
                    status,
                )
            weight_evidence.update(status)
        else:
            try:
                checkpoint = torch.load(status["cache_path"], map_location="cpu", weights_only=True)
            except Exception as error:
                raise UserError(
                    "PRETRAINED_WEIGHTS_CACHE_INVALID",
                    f"无法安全读取 ResNet18 缓存：{type(error).__name__}: {error}",
                    "重新准备经官方来源校验的权重，或选择随机初始化。",
                    status,
                ) from error

    try:
        from torchvision.models import resnet18
    except Exception as error:
        raise UserError(
            "PRESET_DEPENDENCY_MISSING",
            f"ResNet18 需要可用的 torchvision：{type(error).__name__}: {error}",
            "在当前项目 Python 环境中安装与 PyTorch 兼容的 torchvision；Tiny CNN 不依赖 torchvision。",
            {"dependency": "torchvision", "torch_version": str(torch.__version__)},
        ) from error
    model = resnet18(weights=None)
    if checkpoint is not None:
        if not isinstance(checkpoint, dict):
            raise UserError("PRETRAINED_WEIGHTS_CACHE_INVALID", "ResNet18 权重文件不是 state_dict 对象。")
        # Keep the pretrained feature extractor while intentionally resetting
        # the task-specific classifier for the user's class count.
        backbone = {name: value for name, value in checkpoint.items() if not name.startswith("fc.")}
        missing, unexpected = model.load_state_dict(backbone, strict=False)
        if unexpected or set(missing) != {"fc.weight", "fc.bias"}:
            raise UserError(
                "PRETRAINED_WEIGHTS_INCOMPATIBLE",
                "ResNet18 权重与 torchvision 模型结构不匹配。",
                "确认使用官方 IMAGENET1K_V1 checkpoint 和兼容的 torchvision。",
                {"missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected)},
            )
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.model_maven_weight_plan = weight_evidence
    return model


def classification_batch(
    root,
    batch_size=8,
    split="train",
    split_ratios=None,
    seed=0,
    transforms=None,
    sample_offset=None,
    _digest_cache=None,
    _split_cache=None,
):
    """Load one deterministic real-image batch from a class-folder project.

    This composes the shared content index, split, stratified sampler, and
    transform registry. The dataset directory is read in place and never
    rearranged. The same factory is suitable for ``--batch`` and managed try.
    """
    torch = _torch()
    if type(batch_size) is not int or batch_size < 1:
        raise UserError("PRESET_BATCH_SIZE_INVALID", "分类预设 batch_size 必须为正整数。")
    if sample_offset is not None and (type(sample_offset) is not int or sample_offset < 0):
        raise UserError("PRESET_SAMPLE_OFFSET_INVALID", "sample_offset 必须是非负整数。")
    root = Path(root).expanduser().resolve()
    index = scan_class_folder(root, digest_cache=_digest_cache)
    index = load_project_exclusions(Path.cwd(), root, index)
    if split not in {"train", "validation", "test"}:
        raise UserError("DATA_SPLIT_INVALID", "split 必须是 train、validation 或 test。")
    ratios = split_ratios or {"train": 1.0}
    cache_key = None
    if isinstance(_split_cache, dict) and index["sample_count"] <= 100_000:
        try:
            cache_key = json.dumps([index["dataset_fingerprint"], ratios, seed], sort_keys=True,
                                   separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            pass  # Let the normal split validator report unsupported values.
    if cache_key is not None and _split_cache.get("key") == cache_key:
        split_plan = _split_cache["plan"]
    else:
        split_plan = split_classification_index(index, ratios, seed=seed)
        if isinstance(_split_cache, dict):
            _split_cache.clear()
            if cache_key is not None:
                _split_cache.update({"key": cache_key, "plan": split_plan, "epochs": OrderedDict()})
    split_samples = split_plan["splits"].get(split)
    if not split_samples:
        raise UserError("DATA_SPLIT_EMPTY", f"{split} 切分没有图像样本；不会借用其他切分补足。")
    cursor_key = f"{root}|{split}|{seed}"
    start = _CLASSIFICATION_BATCH_CURSORS.get(cursor_key, 0) if sample_offset is None else sample_offset
    epoch_offset = start % len(split_samples)
    actual_batch_size = min(batch_size, len(split_samples) - epoch_offset)
    selected = _classification_window(split_samples, actual_batch_size, seed, start,
                                      _split_cache.get("epochs") if cache_key is not None else None, split)
    try:
        from PIL import Image
    except ImportError as error:
        raise UserError(
            "PRESET_DEPENDENCY_MISSING",
            "分类图像预设需要当前项目 Python 环境中的 Pillow。",
            "安装 Pillow 后重试；目录索引、指纹和切分本身不依赖 Pillow。",
            {"dependency": "Pillow"},
        ) from error
    specifications = transforms if transforms is not None else [
        {"type": "Resize", "size": 32},
        {"type": "ToTensor"},
    ]
    image_tensors = []
    labels = []
    for sample in selected:
        path = (root / sample["path"]).resolve()
        try:
            path.relative_to(root)
            digest = _sha256(path)
            if digest != sample["sha256"]:
                raise UserError(
                    "DATASET_PROTOCOL_INVALIDATED",
                    f"读取样本期间数据内容发生变化：{sample['path']}",
                    "重新扫描并明确建立新的切分协议。",
                    {"path": sample["path"], "expected_sha256": sample["sha256"], "actual_sha256": digest},
                )
            with Image.open(path) as source:
                image = source.convert("RGB")
                transformed = run_transform_pipeline(
                    specifications,
                    image,
                    seed=seed + start,
                )["output"]
        except UserError:
            raise
        except Exception as error:
            raise UserError(
                "CLASS_SAMPLE_LOAD_FAILED",
                f"无法加载分类样本 {sample['path']}：{type(error).__name__}: {error}",
                "修复该样本或在数据协议中明确排除后重试。",
                {"path": sample["path"]},
            ) from error
        if not isinstance(transformed, torch.Tensor) or transformed.ndim != 3:
            raise UserError(
                "PRESET_TRANSFORM_OUTPUT_INVALID",
                "分类变换必须输出 CHW Tensor。",
                "检查 transforms 顺序；通常图像应先 Resize，再 ToTensor，最后 Normalize。",
                {"path": sample["path"]},
            )
        image_tensors.append(transformed)
        labels.append(index["class_to_idx"][sample["label"]])
    try:
        inputs = torch.stack(image_tensors)
    except Exception as error:
        raise UserError(
            "PRESET_BATCH_SHAPE_MISMATCH",
            f"选中图像无法组成同形状批次：{type(error).__name__}: {error}",
            "确认所有变换输出一致的 C/H/W；不会自动裁切或补齐。",
            {"sample_ids": [sample["sample_id"] for sample in selected]},
        ) from error
    selected_ids = [sample["sample_id"] for sample in selected]
    selected_classes = sorted({sample["label"] for sample in selected})
    subset_evidence = {
        "dataset_fingerprint": index["dataset_fingerprint"],
        "split_fingerprint": split_plan["split_fingerprint"],
        "split": split,
        "seed": seed,
        "start_offset": start,
        "sample_ids": selected_ids,
    }
    subset_fingerprint = hashlib.sha256(
        json.dumps(subset_evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "inputs": inputs,
        "targets": torch.tensor(labels, dtype=torch.long),
        "dataset": {
            "dataset_fingerprint": index["dataset_fingerprint"],
            "split_fingerprint": split_plan["split_fingerprint"],
            "subset_fingerprint": subset_fingerprint,
            "sample_ids": selected_ids,
            "class_to_idx": index["class_to_idx"],
            "coverage": {
                "class_count": len(selected_classes),
                "available_class_count": len({sample["label"] for sample in split_samples}),
                "complete_class_coverage": len(selected_classes) == len({sample["label"] for sample in split_samples}),
                "requested_batch_size": batch_size,
                "actual_count": len(selected),
            },
        },
    }
    # Advance only after the complete batch and its evidence are ready. A
    # failed read or transform must not silently skip samples on retry.
    if sample_offset is None:
        _CLASSIFICATION_BATCH_CURSORS[cursor_key] = start + len(selected)
    return result



def classification_step(model, batch, context=None):
    """Native managed-step for the preset's ``(B, C)`` logits contract."""
    torch = _torch()
    loss_inputs = batch.get("loss_inputs") if isinstance(batch, dict) else None
    if isinstance(loss_inputs, dict) and isinstance(batch.get("args"), (list, tuple)):
        inputs = batch["args"][0] if batch["args"] else None
        targets = loss_inputs.get("targets")
        dataset = loss_inputs.get("dataset")
    else:
        inputs = batch.get("inputs") if isinstance(batch, dict) else None
        targets = batch.get("targets") if isinstance(batch, dict) else None
        dataset = batch.get("dataset") if isinstance(batch, dict) else None
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise UserError("PRESET_BATCH_INVALID", "分类批次必须包含 Tensor inputs 和 Tensor targets。")
    if inputs.ndim != 4 or targets.ndim != 1 or inputs.shape[0] != targets.shape[0]:
        raise UserError(
            "PRESET_BATCH_INVALID",
            "分类批次应为 inputs=[B,C,H,W]、targets=[B]。",
            "用 classification_batch 构造预设批次，或保证自定义批次遵守同一形状契约。",
            {"inputs_shape": list(inputs.shape), "targets_shape": list(targets.shape)},
        )
    logits = model(inputs)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != targets.shape[0]:
        shape = list(logits.shape) if isinstance(logits, torch.Tensor) else None
        raise UserError(
            "PRESET_OUTPUT_SHAPE_INVALID",
            "分类模型必须返回 logits=[B, classes]。",
            "分类预设只对分类输出做此约束；自定义任务可提供自己的 training_step。",
            {"actual_shape": shape},
        )
    loss = torch.nn.functional.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return {
        "loss": loss,
        "losses": {"classification": loss.detach()},
        "metrics": {"accuracy": accuracy.detach()},
        "outputs": {"logits": logits},
        "dataset": dataset,
    }


def _torch():
    try:
        import torch
    except ImportError as error:
        raise UserError(
            "PRESET_DEPENDENCY_MISSING",
            "模型预设需要当前项目 Python 环境中的 PyTorch。",
            "选择已有兼容环境，或先准备项目环境。",
            {"dependency": "torch"},
        ) from error
    return torch


def _torch_hub_cache_dir(cache_dir):
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return Path(torch_home).expanduser().resolve() / "hub" / "checkpoints"
    xdg_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_home).expanduser() if xdg_home else Path.home() / ".cache"
    return base.resolve() / "torch" / "hub" / "checkpoints"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification_window(samples, batch_size, seed, start, epoch_cache=None, split=None):
    """Take the next deterministic shuffled minibatch, wrapping by epoch."""
    if not samples:
        return []
    result = []
    shuffled_epochs = {}
    for position in range(start, start + batch_size):
        epoch, offset = divmod(position, len(samples))
        if epoch not in shuffled_epochs:
            cached = epoch_cache.get((split, epoch)) if isinstance(epoch_cache, OrderedDict) else None
            if cached is None:
                epoch_seed = seed + epoch
                cached = sorted(
                    samples,
                    key=lambda sample: (
                        hashlib.sha256(
                            f"model-maven:classification-shuffle:v1:{epoch_seed}:{sample['sample_id']}".encode("utf-8")
                        ).digest(),
                        sample["sample_id"],
                    ),
                )
                if isinstance(epoch_cache, OrderedDict):
                    epoch_cache[(split, epoch)] = cached
                    while len(epoch_cache) > 2:
                        epoch_cache.popitem(last=False)
            elif isinstance(epoch_cache, OrderedDict):
                epoch_cache.move_to_end((split, epoch))
            shuffled_epochs[epoch] = cached
        result.append(shuffled_epochs[epoch][offset])
    return result


def _batch_state_dict():
    return {"version": 1, "cursors": dict(sorted(_CLASSIFICATION_BATCH_CURSORS.items()))}


def _load_batch_state_dict(state):
    if not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("cursors"), dict):
        raise UserError("PRESET_BATCH_STATE_INVALID", "分类预设 batch 状态格式无效。")
    cursors = state["cursors"]
    if any(not isinstance(key, str) or type(value) is not int or value < 0 for key, value in cursors.items()):
        raise UserError("PRESET_BATCH_STATE_INVALID", "分类预设 batch 状态包含无效数据游标。")
    _CLASSIFICATION_BATCH_CURSORS.clear()
    _CLASSIFICATION_BATCH_CURSORS.update(cursors)


classification_batch.state_dict = _batch_state_dict
classification_batch.load_state_dict = _load_batch_state_dict


def graph_image_batch(root, input_name, target_name, batch_size=8, split="train",
                      split_ratios=None, seed=0, transforms=None, split_transforms=None, sample_offset=None):
    """Bind a class-folder image batch to one explicitly named graph input.

    The returned named target is consumed by the graph's managed loss step.
    Data and transforms are selected by the caller; no packages are installed.
    """
    for value, label in ((input_name, "input_name"), (target_name, "target_name")):
        if not isinstance(value, str) or not value or len(value) > 128 or any(ord(c) < 32 for c in value):
            raise UserError("GRAPH_IMAGE_BINDING_INVALID", f"{label} 必须为不超过 128 字符的非空名称。")
    if split_transforms is not None:
        if (transforms is not None or not isinstance(split_transforms, dict)
                or set(split_transforms) - {"train", "validation", "test"}
                or split not in split_transforms or not isinstance(split_transforms[split], list)):
            raise UserError("GRAPH_IMAGE_TRANSFORMS_INVALID", "按切分变换需要为当前切分提供列表，不能同时设置 transforms。")
        transforms = split_transforms[split]
    try:
        resolved_root = str(Path(root).expanduser().resolve())
        identity_bytes = json.dumps({"root": resolved_root, "input_name": input_name, "target_name": target_name,
                                     "batch_size": batch_size, "split": split, "split_ratios": split_ratios,
                                     "seed": seed, "transforms": transforms}, sort_keys=True,
                                    separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OSError) as error:
        raise UserError("GRAPH_IMAGE_CONFIG_INVALID", f"图像批次配置无法记录：{error}") from error
    cursor_key = f"{resolved_root}|{split}|{seed}"
    prior_cursor = _CLASSIFICATION_BATCH_CURSORS.get(cursor_key)
    cache = _GRAPH_IMAGE_INDEX_CACHES.setdefault(resolved_root, {"digests": {}, "split": {}})
    _GRAPH_IMAGE_INDEX_CACHES.move_to_end(resolved_root)
    while len(_GRAPH_IMAGE_INDEX_CACHES) > 2:
        _GRAPH_IMAGE_INDEX_CACHES.popitem(last=False)
    if sample_offset is None:
        _CLASSIFICATION_BATCH_CURSORS[cursor_key] = _GRAPH_IMAGE_CURSORS.get(cursor_key, 0)
    try:
        result = classification_batch(root=root, batch_size=batch_size, split=split,
                                      split_ratios=split_ratios, seed=seed,
                                      transforms=transforms, sample_offset=sample_offset,
                                      _digest_cache=cache["digests"], _split_cache=cache["split"])
        next_cursor = _CLASSIFICATION_BATCH_CURSORS.get(cursor_key)
    finally:
        if sample_offset is None:
            if prior_cursor is None:
                _CLASSIFICATION_BATCH_CURSORS.pop(cursor_key, None)
            else:
                _CLASSIFICATION_BATCH_CURSORS[cursor_key] = prior_cursor
    if sample_offset is None:
        evidence = {"config": hashlib.sha256(identity_bytes).hexdigest(),
                    "dataset": result["dataset"]["dataset_fingerprint"],
                    "split": result["dataset"]["split_fingerprint"]}
        previous = _GRAPH_IMAGE_EVIDENCE.get(cursor_key)
        if previous is not None and previous != evidence:
            raise UserError("GRAPH_IMAGE_STATE_CHANGED", "图像数据、切分或批次配置与已保存游标不一致。",
                            "检查数据与配置；如需使用新数据，显式开始新运行而不是继续旧游标。")
        _GRAPH_IMAGE_EVIDENCE[cursor_key] = evidence
        _GRAPH_IMAGE_CURSORS[cursor_key] = next_cursor
    dataset = {**result["dataset"], "split": split}
    return {"args": [], "kwargs": {input_name: result["inputs"]},
            "loss_inputs": {"targets": {target_name: result["targets"]}, "dataset": dataset}}


def _graph_image_state_dict():
    return {"version": 1, "cursors": dict(sorted(_GRAPH_IMAGE_CURSORS.items())),
            "evidence": {key: dict(value) for key, value in sorted(_GRAPH_IMAGE_EVIDENCE.items())}}


def _graph_image_load_state_dict(state):
    if (not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("cursors"), dict)
            or not isinstance(state.get("evidence"), dict) or set(state["cursors"]) != set(state["evidence"])):
        raise UserError("GRAPH_IMAGE_STATE_INVALID", "图像批次恢复状态缺少游标或数据身份。")
    for key, cursor in state["cursors"].items():
        record = state["evidence"].get(key)
        if (not isinstance(key, str) or type(cursor) is not int or cursor < 0 or not isinstance(record, dict)
                or set(record) != {"config", "dataset", "split"}
                or any(not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
                       for value in record.values())):
            raise UserError("GRAPH_IMAGE_STATE_INVALID", "图像批次恢复状态包含无效身份或游标。")
    _GRAPH_IMAGE_CURSORS.clear()
    _GRAPH_IMAGE_CURSORS.update(state["cursors"])
    _GRAPH_IMAGE_EVIDENCE.clear()
    _GRAPH_IMAGE_EVIDENCE.update({key: dict(value) for key, value in state["evidence"].items()})
    # Recovery always re-hashes on its first batch; a prior process's metadata
    # cache is neither durable nor proof of current dataset contents.
    _GRAPH_IMAGE_INDEX_CACHES.clear()


graph_image_batch.state_dict = _graph_image_state_dict
graph_image_batch.load_state_dict = _graph_image_load_state_dict


def _validate_model_dimensions(num_classes, in_channels, width):
    for name, value, minimum in (
        ("num_classes", num_classes, 2),
        ("in_channels", in_channels, 1),
        ("width", width, 1),
    ):
        if type(value) is not int or value < minimum:
            raise UserError("PRESET_MODEL_PARAMETER_INVALID", f"{name} 必须是大于等于 {minimum} 的整数。")


