"""Ordered, inspectable data transforms with per-step port contracts."""

import importlib
import random

from ..protocol import UserError


TRANSFORM_REGISTRY_VERSION = 1

_BUILTINS = {
    "Resize": {
        "description": "Resize an image or CHW/BCHW tensor while preserving its value type.",
        "fields": {
            "size": {"type": "integer|pair", "required": True, "help": "Output spatial size (height, width)."},
            "interpolation": {"type": "string", "required": False, "default": "bilinear", "help": "Tensor interpolation mode; PIL uses its bilinear resampler."},
        },
    },
    "RandomResizedCrop": {
        "description": "Randomly crop a square fraction of the input and resize it to the requested output size.",
        "fields": {
            "size": {"type": "integer|pair", "required": True, "help": "Output spatial size (height, width)."},
            "scale": {"type": "number-pair", "required": True, "help": "Random crop area fraction, 0 < min <= max <= 1."},
        },
    },
    "ToTensor": {
        "description": "Convert a PIL/NumPy image to a CHW tensor; uint8 values become float32 in [0, 1].",
        "fields": {},
    },
    "Normalize": {
        "description": "Normalize a CHW/BCHW tensor by channel mean and standard deviation.",
        "fields": {
            "mean": {"type": "number|number[]", "required": True, "help": "Per-channel mean."},
            "std": {"type": "number|number[]", "required": True, "help": "Per-channel standard deviation; every value must be positive."},
        },
    },
    "RandomHorizontalFlip": {
        "description": "Flip an image horizontally with a seeded probability.",
        "fields": {
            "p": {"type": "number", "required": False, "default": 0.5, "help": "Probability from 0 through 1."},
        },
    },
}

_CONTRACT_FIELDS = {"kind", "shape", "dtype", "layout", "memory_format"}


def describe_transform_registry():
    """Return form/completion metadata without importing image or tensor libraries."""
    return {
        "version": TRANSFORM_REGISTRY_VERSION,
        "transforms": {
            name: {"description": entry["description"], "fields": entry["fields"]}
            for name, entry in _BUILTINS.items()
        },
        "custom": {
            "entrypoint": "module:callable",
            "parameters": "JSON object passed to the callable as keyword arguments",
            "ports": ["input_contract", "output_contract"],
        },
    }


def validate_transform_pipeline(specifications):
    """Validate transform definitions without reordering or executing them."""
    if not isinstance(specifications, list):
        raise UserError("TRANSFORM_PIPELINE_INVALID", "transforms 必须是按执行顺序排列的数组。")
    validated = []
    for index, raw in enumerate(specifications):
        location = f"transforms[{index}]"
        if not isinstance(raw, dict):
            raise _definition_error(location, "每一步必须是对象。")
        kind = raw.get("type")
        if not isinstance(kind, str) or not kind:
            raise _definition_error(location, "type 必须是非空字符串。")
        if kind == "custom":
            allowed = {"type", "entrypoint", "parameters", "input_contract", "output_contract"}
            unknown = sorted(set(raw) - allowed)
            if unknown:
                raise _definition_error(location, f"自定义步骤含未知字段：{', '.join(unknown)}。")
            entrypoint = raw.get("entrypoint")
            if not isinstance(entrypoint, str) or ":" not in entrypoint:
                raise _definition_error(location, "custom.entrypoint 必须是 module:callable。")
            parameters = raw.get("parameters", {})
            if not isinstance(parameters, dict):
                raise _definition_error(location, "custom.parameters 必须是对象。")
        elif kind in _BUILTINS:
            allowed = {"type", "input_contract", "output_contract", *_BUILTINS[kind]["fields"]}
            unknown = sorted(set(raw) - allowed)
            if unknown:
                raise _definition_error(location, f"{kind} 含未知字段：{', '.join(unknown)}。")
            parameters = {key: value for key, value in raw.items() if key not in {"type", "input_contract", "output_contract"}}
            _validate_builtin_parameters(kind, parameters, location)
        else:
            raise _definition_error(location, f"未知变换类型 {kind!r}。", {"available": sorted([*_BUILTINS, "custom"])})
        input_contract = _validate_port_contract(raw.get("input_contract"), f"{location}.input_contract")
        output_contract = _validate_port_contract(raw.get("output_contract"), f"{location}.output_contract")
        validated.append({
            "type": kind,
            "parameters": dict(parameters),
            "entrypoint": raw.get("entrypoint"),
            "input_contract": input_contract,
            "output_contract": output_contract,
        })
    return validated


def run_transform_pipeline(specifications, sample, resolve_entrypoint=None, seed=0):
    """Run steps exactly as declared and report the actual ports at every boundary."""
    if type(seed) is not int:
        raise UserError("TRANSFORM_SEED_INVALID", "变换种子必须是整数。")
    steps = validate_transform_pipeline(specifications)
    current = sample
    observations = []
    for index, step in enumerate(steps):
        name = step["type"]
        input_port = describe_value(current)
        try:
            _check_port_contract(step["input_contract"], input_port, index, name, "input")
            transform = _build_transform(step, resolve_entrypoint)
            current = transform(current, random.Random(_step_seed(seed, index)))
            output_port = describe_value(current)
            _check_port_contract(step["output_contract"], output_port, index, name, "output")
        except UserError:
            raise
        except Exception as error:
            raise UserError(
                "TRANSFORM_STEP_FAILED",
                f"第 {index + 1} 步 {name} 执行失败：{type(error).__name__}: {error}",
                "检查该步骤的输入类型、维度和参数；流水线保持声明顺序，不会自动重排。",
                {"step": index, "type": name, "input": input_port},
            ) from error
        observations.append({
            "index": index,
            "type": name,
            "input": input_port,
            "output": output_port,
            "status": "passed",
        })
    return {
        "registry_version": TRANSFORM_REGISTRY_VERSION,
        "seed": seed,
        "step_count": len(steps),
        "steps": observations,
        "output": current,
        "output_port": describe_value(current),
    }


def describe_value(value):
    """Describe value structure and tensor/image ports without serializing payloads."""
    kind = _value_kind(value)
    result = {"kind": kind, "type": type(value).__name__}
    shape = _shape(value, kind)
    if shape is not None:
        result["shape"] = list(shape)
        result["rank"] = len(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        result["dtype"] = str(dtype).replace("torch.", "").replace("numpy.", "")
    elif kind == "image":
        result["dtype"] = "uint8"
    layout = getattr(value, "layout", None)
    if layout is not None:
        result["layout"] = str(layout).replace("torch.", "")
        if kind == "tensor":
            result["memory_format"] = _memory_format(value)
    elif kind == "image":
        result["layout"] = "HWC"
    elif kind == "numpy":
        result["layout"] = "array"
    return result


def _build_transform(step, resolve_entrypoint):
    kind = step["type"]
    params = step["parameters"]
    if kind == "custom":
        resolver = resolve_entrypoint or _resolve_entrypoint
        try:
            transform = resolver(step["entrypoint"])(**params)
        except UserError:
            raise
        except Exception as error:
            raise UserError(
                "TRANSFORM_CONSTRUCTION_FAILED",
                f"无法构造自定义变换 {step['entrypoint']}：{type(error).__name__}: {error}",
                "确认 module:callable 可导入且 parameters 与工厂签名一致。",
                {"entrypoint": step["entrypoint"]},
            ) from error
        if not callable(transform):
            raise UserError("TRANSFORM_CONSTRUCTION_FAILED", "自定义变换工厂必须返回可调用对象。", "检查 factory 的返回值。")
        return lambda value, _rng: transform(value)
    if kind == "Resize":
        return lambda value, _rng: _resize(value, params["size"], params.get("interpolation", "bilinear"))
    if kind == "RandomResizedCrop":
        return lambda value, rng: _random_resized_crop(value, params["size"], params["scale"], rng)
    if kind == "ToTensor":
        return lambda value, _rng: _to_tensor(value)
    if kind == "Normalize":
        return lambda value, _rng: _normalize(value, params["mean"], params["std"])
    if kind == "RandomHorizontalFlip":
        return lambda value, rng: _horizontal_flip(value, params.get("p", 0.5), rng)
    raise AssertionError(f"validated transform has no builder: {kind}")


def _resize(value, size, interpolation):
    height, width = _size_pair(size)
    kind = _value_kind(value)
    if kind == "image":
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image)
        modes = {
            "bilinear": "BILINEAR",
            "nearest": "NEAREST",
            "bicubic": "BICUBIC",
        }
        mode = getattr(resampling, modes.get(interpolation, ""), None)
        if mode is None:
            raise ValueError(f"unsupported PIL interpolation {interpolation!r}")
        return value.resize((width, height), resample=mode)
    if kind != "tensor":
        raise TypeError(f"Resize expects a PIL image or tensor, received {kind}")
    if value.ndim not in {3, 4}:
        raise ValueError(f"Resize expects CHW or BCHW tensor, received rank {value.ndim}")
    import torch.nn.functional as functional

    if interpolation not in {"bilinear", "nearest", "bicubic", "area"}:
        raise ValueError(f"unsupported tensor interpolation {interpolation!r}")
    input_dtype = value.dtype
    tensor = value if value.is_floating_point() else value.float()
    kwargs = {"size": (height, width), "mode": interpolation}
    if interpolation in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    result = functional.interpolate(tensor.unsqueeze(0) if tensor.ndim == 3 else tensor, **kwargs)
    if value.ndim == 3:
        result = result.squeeze(0)
    if not value.is_floating_point():
        result = result.round().to(input_dtype)
    return result


def _random_resized_crop(value, size, scale, rng):
    if not isinstance(scale, (list, tuple)) or len(scale) != 2:
        raise ValueError("scale must contain min and max crop fractions")
    minimum, maximum = float(scale[0]), float(scale[1])
    if not (0 < minimum <= maximum <= 1):
        raise ValueError("scale must satisfy 0 < min <= max <= 1")
    height, width = _size_pair(size)
    fraction = rng.uniform(minimum, maximum)
    kind = _value_kind(value)
    if kind == "image":
        source_width, source_height = value.size
        crop_width = max(1, min(source_width, round(source_width * fraction ** 0.5)))
        crop_height = max(1, min(source_height, round(source_height * fraction ** 0.5)))
        left = rng.randrange(source_width - crop_width + 1)
        top = rng.randrange(source_height - crop_height + 1)
        crop = value.crop((left, top, left + crop_width, top + crop_height))
        return _resize(crop, [height, width], "bilinear")
    if kind != "tensor" or value.ndim not in {3, 4}:
        raise TypeError("RandomResizedCrop expects a PIL image or CHW/BCHW tensor")
    source_height, source_width = value.shape[-2:]
    crop_width = max(1, min(source_width, round(source_width * fraction ** 0.5)))
    crop_height = max(1, min(source_height, round(source_height * fraction ** 0.5)))
    left = rng.randrange(source_width - crop_width + 1)
    top = rng.randrange(source_height - crop_height + 1)
    crop = value[..., top:top + crop_height, left:left + crop_width]
    return _resize(crop, [height, width], "bilinear")


def _to_tensor(value):
    if _value_kind(value) == "tensor":
        return value
    if _value_kind(value) == "image":
        import numpy as np

        array = np.asarray(value)
    elif _value_kind(value) == "numpy":
        array = value
    else:
        raise TypeError(f"ToTensor expects a PIL image, NumPy array, or tensor, received {_value_kind(value)}")
    import torch

    tensor = torch.from_numpy(array.copy())
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3:
        raise ValueError(f"ToTensor expects a 2D or 3D image array, received rank {tensor.ndim}")
    tensor = tensor.permute(2, 0, 1).contiguous()
    if tensor.dtype == torch.uint8:
        return tensor.to(dtype=torch.float32).div(255)
    return tensor


def _normalize(value, mean, std):
    if _value_kind(value) != "tensor":
        raise TypeError(f"Normalize expects a tensor, received {_value_kind(value)}")
    if value.ndim not in {3, 4}:
        raise ValueError(f"Normalize expects CHW or BCHW tensor, received rank {value.ndim}")
    if not value.is_floating_point():
        raise TypeError(f"Normalize expects a floating-point tensor, received {value.dtype}")
    channels = value.shape[-3]
    means = _channel_values(mean, channels, "mean")
    stds = _channel_values(std, channels, "std")
    if any(item <= 0 for item in stds):
        raise ValueError("std values must be positive")
    shape = (1,) * (value.ndim - 3) + (channels, 1, 1)
    mean_tensor = value.new_tensor(means).reshape(shape)
    std_tensor = value.new_tensor(stds).reshape(shape)
    return (value - mean_tensor) / std_tensor


def _horizontal_flip(value, probability, rng):
    if rng.random() >= probability:
        return value
    kind = _value_kind(value)
    if kind == "image":
        from PIL import Image

        transpose = getattr(Image, "Transpose", Image)
        return value.transpose(transpose.FLIP_LEFT_RIGHT)
    if kind == "tensor":
        if value.ndim not in {3, 4}:
            raise ValueError(f"RandomHorizontalFlip expects CHW/BCHW tensor, received rank {value.ndim}")
        return value.flip(-1)
    if kind == "numpy":
        return value[..., ::-1].copy()
    raise TypeError(f"RandomHorizontalFlip expects an image, tensor, or NumPy array, received {kind}")


def _validate_builtin_parameters(kind, params, location):
    if kind == "Resize":
        if "size" not in params:
            raise _definition_error(location, "Resize.size 为必填项。")
        try:
            _size_pair(params["size"])
        except (TypeError, ValueError) as error:
            raise _definition_error(location, f"Resize.size 无效：{error}") from error
        if params.get("interpolation", "bilinear") not in {"bilinear", "nearest", "bicubic", "area"}:
            raise _definition_error(location, "Resize.interpolation 必须是 bilinear、nearest、bicubic 或 area。")
    elif kind == "RandomResizedCrop":
        if "size" not in params or "scale" not in params:
            raise _definition_error(location, "RandomResizedCrop.size 和 RandomResizedCrop.scale 都是必填项。")
        try:
            _size_pair(params["size"])
        except (TypeError, ValueError) as error:
            raise _definition_error(location, f"RandomResizedCrop.size 无效：{error}") from error
        scale = params["scale"]
        if not isinstance(scale, (list, tuple)) or len(scale) != 2:
            raise _definition_error(location, "RandomResizedCrop.scale 必须是 [min, max]。")
        minimum, maximum = scale
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in scale) or not 0 < minimum <= maximum <= 1:
            raise _definition_error(location, "RandomResizedCrop.scale 必须满足 0 < min <= max <= 1。")
    elif kind == "ToTensor" and params:
        raise _definition_error(location, "ToTensor 不接受参数。")
    elif kind == "Normalize":
        if "mean" not in params or "std" not in params:
            raise _definition_error(location, "Normalize.mean 和 Normalize.std 都是必填项。")
        try:
            for field in ("mean", "std"):
                _channel_values(params[field], None, field)
            std_values = _channel_values(params["std"], None, "std")
        except ValueError as error:
            raise _definition_error(location, str(error)) from error
        if any(value <= 0 for value in std_values):
            raise _definition_error(location, "Normalize.std 的每个值都必须大于 0。")
    elif kind == "RandomHorizontalFlip":
        probability = params.get("p", 0.5)
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise _definition_error(location, "RandomHorizontalFlip.p 必须在 0 到 1 之间。")


def _validate_port_contract(contract, location):
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise _definition_error(location, "必须是对象。")
    unknown = sorted(set(contract) - _CONTRACT_FIELDS)
    if unknown:
        raise _definition_error(location, f"包含未知字段：{', '.join(unknown)}。")
    supported_kinds = {"tensor", "image", "numpy", "dict", "tuple", "list", "scalar", "object", "any"}
    if "kind" in contract and contract["kind"] not in supported_kinds:
        raise _definition_error(location, f"kind 必须是 {', '.join(sorted(supported_kinds))} 之一。")
    if "shape" in contract:
        shape = contract["shape"]
        if not isinstance(shape, list) or any(dim is not None and (type(dim) is not int or dim < 0) for dim in shape):
            raise _definition_error(location, "shape 必须是由非负整数或 null 组成的数组。")
    for field in ("dtype", "layout", "memory_format"):
        if field in contract and (not isinstance(contract[field], str) or not contract[field]):
            raise _definition_error(location, f"{field} 必须是非空字符串。")
    if "memory_format" in contract and contract["memory_format"] not in {"contiguous", "channels_last", "non_contiguous"}:
        raise _definition_error(location, "memory_format 必须是 contiguous、channels_last 或 non_contiguous。")
    return dict(contract)


def _check_port_contract(contract, actual, index, name, direction):
    if contract is None:
        return
    mismatches = {}
    for field, expected in contract.items():
        observed = actual.get(field)
        if field == "shape":
            matches = isinstance(observed, list) and len(observed) == len(expected) and all(
                target is None or target == got for target, got in zip(expected, observed)
            )
        elif expected == "any" and field == "kind":
            matches = True
        else:
            matches = expected == observed
        if not matches:
            mismatches[field] = {"expected": expected, "actual": observed}
    if mismatches:
        raise UserError(
            "TRANSFORM_PORT_MISMATCH",
            f"第 {index + 1} 步 {name} 的 {direction} 端口不符合声明。",
            "检查前一步输出、当前步骤参数或该步骤的输入/输出契约。",
            {"step": index, "type": name, "port": direction, "mismatches": mismatches, "actual": actual},
        )


def _shape(value, kind):
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            return None
    if kind == "image":
        width, height = value.size
        bands = len(value.getbands())
        return (height, width, bands) if bands > 1 else (height, width)
    return None


def _memory_format(value):
    if value.ndim == 4:
        try:
            import torch

            if value.is_contiguous(memory_format=torch.channels_last):
                return "channels_last"
        except (ImportError, TypeError):
            pass
    try:
        return "contiguous" if value.is_contiguous() else "non_contiguous"
    except (AttributeError, TypeError):
        return "unknown"


def _value_kind(value):
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
        return "tensor"
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return "numpy"
    if hasattr(value, "size") and hasattr(value, "getbands"):
        return "image"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return "scalar"
    return "object"


def _size_pair(size):
    if type(size) is int and size > 0:
        return size, size
    if isinstance(size, (list, tuple)) and len(size) == 2 and all(type(value) is int and value > 0 for value in size):
        return tuple(size)
    raise ValueError("expected a positive integer or a pair of positive integers")


def _channel_values(value, channels, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number or non-empty numeric list")
    if isinstance(value, (int, float)):
        values = [float(value)]
    elif isinstance(value, (list, tuple)) and value and all(
        not isinstance(item, bool) and isinstance(item, (int, float)) for item in value
    ):
        values = [float(item) for item in value]
    else:
        raise ValueError(f"{field} must be a number or non-empty numeric list")
    if channels is not None and len(values) not in {1, channels}:
        raise ValueError(f"{field} has {len(values)} values for {channels} channels")
    return values * channels if channels is not None and len(values) == 1 else values


def _step_seed(seed, index):
    # Python's hash is process-randomized; arithmetic keeps replay stable across processes.
    return (seed + 0x9E3779B97F4A7C15 * (index + 1)) & ((1 << 64) - 1)


def _resolve_entrypoint(reference):
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise UserError("TRANSFORM_ENTRYPOINT_INVALID", "自定义变换入口必须使用 module:callable 格式。")
    value = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def _definition_error(location, message, data=None):
    return UserError(
        "TRANSFORM_DEFINITION_INVALID",
        f"{location}: {message}",
        "查看变换注册表字段说明，并保留用户声明的步骤顺序。",
        {"location": location, **(data or {})},
    )
