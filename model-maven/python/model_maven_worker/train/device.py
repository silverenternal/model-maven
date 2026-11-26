"""Explicit single-device placement, nested batch movement and resource reports."""

import re
import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..protocol import UserError
from ..model.structure import registered_storage_key


@dataclass(frozen=True)
class DeviceSelection:
    device: Any
    label: str
    backend: str
    name: str
    requested: str

    def report(self):
        return {
            "device": self.label,
            "device_requested": self.requested,
            "device_backend": self.backend,
            "device_name": self.name,
        }


def resolve_device(torch_module, requested=None):
    """Resolve a requested device using the selected PyTorch runtime."""
    if requested is None:
        requested = "cpu"
    if not isinstance(requested, str):
        raise UserError("DEVICE_INVALID", "设备必须是 PyTorch 设备标识或 auto。")
    requested = requested.strip().lower()
    if requested == "auto":
        cuda = getattr(torch_module, "cuda", None)
        try:
            gpu_available = cuda is not None and cuda.is_available() and cuda.device_count() > 0
        except Exception as error:
            raise UserError(
                "DEVICE_QUERY_FAILED",
                f"自动设备选择无法读取 PyTorch GPU 状态：{type(error).__name__}: {error}",
                "检查当前 Python 环境与 GPU 运行时；显式选择 --device cpu 可继续 CPU 训练。",
            ) from error
        if gpu_available:
            try:
                index = cuda.current_device()
            except Exception as error:
                raise UserError(
                    "DEVICE_QUERY_FAILED",
                    f"自动设备选择无法读取 PyTorch 当前 GPU：{type(error).__name__}: {error}",
                    "检查当前 Python 环境与 GPU 运行时，或显式选择设备。",
                ) from error
            selection = _resolve_explicit_gpu(torch_module, f"cuda:{index}", requested_by="auto")
            return selection
        accelerator = getattr(torch_module, "accelerator", None)
        current = getattr(accelerator, "current_accelerator", None)
        if callable(current):
            try:
                selected = current(check_available=True)
            except Exception as error:
                raise UserError("DEVICE_QUERY_FAILED", f"自动设备选择无法读取加速器状态：{type(error).__name__}: {error}",
                                "检查当前 PyTorch 后端，或显式选择设备。") from error
            if selected is not None and getattr(selected, "type", None) not in {None, "cpu", "meta"}:
                resolved = resolve_device(torch_module, str(selected))
                return DeviceSelection(resolved.device, resolved.label, resolved.backend, resolved.name, "auto")
        device = torch_module.device("cpu")
        return DeviceSelection(device, "cpu", "cpu", "CPU", "auto")
    if requested == "cpu":
        device = torch_module.device("cpu")
        return DeviceSelection(device, "cpu", "cpu", "CPU", "cpu")

    match = re.fullmatch(r"cuda(?::([0-9]+))?", requested)
    if match:
        return _resolve_explicit_gpu(torch_module, requested, requested_by=requested)
    try:
        device = torch_module.device(requested)
    except (RuntimeError, ValueError, TypeError) as error:
        raise UserError("DEVICE_INVALID", f"当前 PyTorch 无法解析设备 {requested!r}：{error}",
                        "输入当前 Python 环境支持的 PyTorch 设备标识。") from error
    if device.type == "meta":
        raise UserError("DEVICE_INVALID", "meta 设备不保存真实参数，不能用于实际训练。")
    provider = getattr(torch_module, device.type, None)
    available = getattr(provider, "is_available", None)
    if callable(available):
        try:
            if not available():
                raise UserError("DEVICE_UNAVAILABLE", f"当前 PyTorch 报告设备 {requested!r} 不可用。",
                                "检查设备、驱动和所选 Python 环境；训练未启动。")
        except UserError:
            raise
        except Exception as error:
            raise UserError("DEVICE_QUERY_FAILED", f"查询设备 {requested!r} 失败：{type(error).__name__}: {error}") from error
    # The backend may be supplied by an installed PyTorch extension. Do not
    # assume it has CUDA's APIs or claim its memory counters are available.
    return DeviceSelection(device, str(device), device.type, str(device), requested)


def _resolve_explicit_gpu(torch_module, requested, requested_by):
    match = re.fullmatch(r"cuda(?::([0-9]+))?", requested)
    if not match:
        raise UserError("DEVICE_INVALID", "首版 GPU 设备必须使用 cuda 或 cuda:N。")
    cuda = getattr(torch_module, "cuda", None)
    try:
        available = cuda is not None and cuda.is_available()
        count = cuda.device_count() if available else 0
    except Exception as error:
        raise UserError(
            "DEVICE_QUERY_FAILED",
            f"无法查询 GPU 运行时状态：{type(error).__name__}: {error}",
            "检查当前 Python 环境、PyTorch 构建和设备驱动，或选择其他设备。",
            {"requested_device": requested},
        ) from error
    if not available:
        raise UserError(
            "DEVICE_UNAVAILABLE",
            "请求了 GPU，但当前 Python/PyTorch 环境没有可用的 CUDA/ROCm 设备。",
            "检查当前解释器的 PyTorch 构建、GPU 驱动和设备可见性；训练未启动。",
            {"requested_device": requested, "visible_device_count": 0},
        )
    if count < 1:
        raise UserError(
            "DEVICE_UNAVAILABLE",
            "PyTorch 报告 GPU 可用，但没有可见设备；训练未启动。",
            "检查 CUDA_VISIBLE_DEVICES 或 ROCm 设备权限。",
            {"requested_device": requested, "visible_device_count": count},
        )
    if match.group(1) is None:
        if count != 1:
            raise UserError(
                "DEVICE_AMBIGUOUS",
                f"当前有 {count} 个可见 GPU；请明确选择一个设备。",
                "例如使用 --device cuda:0。Model-Maven 首版每个运行只使用一个 GPU。",
                {"requested_device": requested, "visible_device_count": count},
            )
        index = 0
    else:
        index = int(match.group(1))
    if index >= count:
        raise UserError(
            "DEVICE_UNAVAILABLE",
            f"GPU cuda:{index} 不在当前可见设备范围内（共 {count} 个）。",
            "查看当前 Python 环境可见的 GPU 后重新选择；训练未启动。",
            {"requested_device": requested, "visible_device_count": count},
        )
    device = torch_module.device(f"cuda:{index}")
    try:
        cuda.set_device(device)
        name = cuda.get_device_name(device)
    except Exception as error:
        raise UserError(
            "DEVICE_UNAVAILABLE",
            f"无法初始化 GPU {device}：{type(error).__name__}: {error}",
            "检查驱动、PyTorch 构建与设备状态；训练未启动。",
            {"requested_device": requested, "device": str(device)},
        ) from error
    backend = "rocm" if getattr(getattr(torch_module, "version", None), "hip", None) else "cuda"
    return DeviceSelection(device, str(device), backend, str(name), requested_by)


def _move_strict_batch(torch_module, value, device):
    """Move supported trees without silently breaking repeated object identities."""
    memo, active, storages = {}, set(), {}
    visits = 0

    def walk(item, depth):
        nonlocal visits
        visits += 1
        if visits > 100_000 or depth > 128:
            raise UserError("BATCH_DEVICE_STRUCTURE_LIMIT", "批次结构超过严格迁移的遍历限制。", "选择 preserve 并由用户步骤处理设备。")
        if not isinstance(item, (torch_module.Tensor, Mapping, list, tuple)):
            return item
        identity = id(item)
        if identity in active:
            raise UserError("BATCH_DEVICE_CYCLE_UNSUPPORTED", "严格迁移不支持包含循环引用的批次。", "选择 preserve 并由用户步骤处理设备。")
        if identity in memo:
            return memo[identity]
        if isinstance(item, torch_module.Tensor):
            result = item.to(device)
            key = registered_storage_key(item)
            if key is not None:
                previous = storages.get(key)
                if previous is not None and previous[0] != identity and (previous[1] or result is not item):
                    raise UserError("BATCH_DEVICE_SHARED_STORAGE_UNSUPPORTED",
                                    "严格迁移无法保证不同 Tensor 之间共享存储的视图关系。",
                                    "选择 preserve，并在用户步骤中显式处理共享存储与设备。")
                storages[key] = (identity, result is not item or (previous is not None and previous[1]))
            memo[identity] = result
            return result
        if isinstance(item, Mapping) and type(item) not in (dict, OrderedDict):
            raise UserError("BATCH_DEVICE_MAPPING_UNSUPPORTED", f"无法确认自定义映射 {type(item).__qualname__} 的状态能被保留。", "选择 preserve 并由用户步骤处理设备。")
        if isinstance(item, list) and type(item) is not list:
            raise UserError("BATCH_DEVICE_CONTAINER_UNSUPPORTED", f"无法自动保留列表子类 {type(item).__qualname__}。", "选择 preserve 并由用户步骤处理设备。")
        if isinstance(item, tuple) and type(item) is not tuple and not hasattr(item, "_fields"):
            raise UserError("BATCH_DEVICE_CONTAINER_UNSUPPORTED", f"无法自动保留元组子类 {type(item).__qualname__}。", "选择 preserve 并由用户步骤处理设备。")
        active.add(identity)
        try:
            if isinstance(item, Mapping):
                entries = [(key, walk(child, depth + 1)) for key, child in item.items()]
                result = OrderedDict(entries) if type(item) is OrderedDict else dict(entries)
            elif isinstance(item, list):
                result = [walk(child, depth + 1) for child in item]
            else:
                values = tuple(walk(child, depth + 1) for child in item)
                if hasattr(item, "_fields"):
                    try:
                        result = type(item)(*values)
                    except (TypeError, ValueError) as error:
                        raise UserError("BATCH_DEVICE_CONTAINER_UNSUPPORTED", f"无法重建具名元组 {type(item).__qualname__}：{error}", "选择 preserve 并由用户步骤处理设备。") from error
                    if type(result) is not type(item):
                        raise UserError("BATCH_DEVICE_CONTAINER_UNSUPPORTED", "具名元组工厂改变了返回类型。", "选择 preserve 并由用户步骤处理设备。")
                else:
                    result = values
            memo[identity] = result
            return result
        finally:
            active.remove(identity)

    return walk(value, 0)


def move_nested_batch(torch_module, value, device, *, strict=False):
    """Move Tensor leaves in mappings and built-in sequences; retain other values."""
    if strict:
        return _move_strict_batch(torch_module, value, device)
    if isinstance(value, torch_module.Tensor):
        return value.to(device)
    if isinstance(value, OrderedDict):
        return type(value)((key, move_nested_batch(torch_module, item, device, strict=strict)) for key, item in value.items())
    if isinstance(value, Mapping):
        moved = {key: move_nested_batch(torch_module, item, device, strict=strict) for key, item in value.items()}
        if type(value) is dict:
            return moved
        try:
            return type(value)(moved)
        except (TypeError, ValueError):
            return moved
    if isinstance(value, list):
        moved = [move_nested_batch(torch_module, item, device, strict=strict) for item in value]
        return moved
    if isinstance(value, tuple):
        moved = tuple(move_nested_batch(torch_module, item, device, strict=strict) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*moved)
        return moved
    return value


def reset_peak_memory(torch_module, selection):
    if selection.device.type == "cuda":
        torch_module.cuda.reset_peak_memory_stats(selection.device)


def resource_snapshot(torch_module, selection):
    """Return measured process/GPU memory; unavailable counters remain null."""
    try:
        import resource

        process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            process_peak *= 1024
        process_peak = int(process_peak)
    except (ImportError, OSError, AttributeError):
        process_peak = None
    report = {"process_peak_rss_bytes": process_peak}
    if selection.device.type != "cuda":
        report.update({
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "free_bytes": None,
            "total_bytes": None,
        })
        if selection.device.type == "cpu":
            return report
        accelerator = getattr(torch_module, "accelerator", None)
        memory = getattr(accelerator, "memory", None)
        current = getattr(accelerator, "current_accelerator", None)
        if not callable(current) or memory is None:
            report["measurement_error"] = "generic accelerator memory API unavailable"
            return report
        try:
            selected = current()
            if selected is None or selected.type != selection.device.type:
                report["measurement_error"] = "selected device does not match current accelerator"
                return report
        except Exception as error:
            report["measurement_error"] = f"accelerator query: {type(error).__name__}: {error}"
            return report
        errors = []
        counters = {
            "allocated_bytes": "memory_allocated",
            "reserved_bytes": "memory_reserved",
            "peak_allocated_bytes": "max_memory_allocated",
            "peak_reserved_bytes": "max_memory_reserved",
        }
        for field, function_name in counters.items():
            function = getattr(memory, function_name, None)
            if not callable(function):
                continue
            try:
                value = int(function(selection.device))
                if value < 0:
                    raise ValueError("negative memory counter")
                report[field] = value
            except Exception as error:
                errors.append(f"{function_name}: {type(error).__name__}: {error}")
        memory_info = getattr(memory, "get_memory_info", None)
        if callable(memory_info):
            try:
                free, total = memory_info(selection.device)
                free, total = int(free), int(total)
                if free < 0 or total < 0 or free > total:
                    raise ValueError("invalid free/total memory")
                report["free_bytes"], report["total_bytes"] = free, total
            except Exception as error:
                errors.append(f"get_memory_info: {type(error).__name__}: {error}")
        report["memory_source"] = "torch.accelerator.memory"
        report["peak_scope"] = "process_lifetime_not_reset"
        if errors:
            report["measurement_error"] = "; ".join(errors)[:1000]
        return report
    cuda = torch_module.cuda
    try:
        cuda.synchronize(selection.device)
        free_bytes, total_bytes = cuda.mem_get_info(selection.device)
        report.update({
            "allocated_bytes": int(cuda.memory_allocated(selection.device)),
            "reserved_bytes": int(cuda.memory_reserved(selection.device)),
            "peak_allocated_bytes": int(cuda.max_memory_allocated(selection.device)),
            "peak_reserved_bytes": int(cuda.max_memory_reserved(selection.device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        })
    except Exception as error:
        report.update({
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "free_bytes": None,
            "total_bytes": None,
            "measurement_error": f"{type(error).__name__}: {error}",
        })
    return report


def is_out_of_memory(torch_module, error):
    oom_type = getattr(torch_module, "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(error, oom_type):
        return True
    cuda = getattr(torch_module, "cuda", None)
    cuda_oom_type = getattr(cuda, "OutOfMemoryError", None)
    if isinstance(cuda_oom_type, type) and isinstance(error, cuda_oom_type):
        return True
    message = str(error).casefold()
    return "out of memory" in message and any(
        marker in message for marker in ("cuda", "hip", "gpu", "alloc")
    )


def out_of_memory_error(torch_module, selection, phase, error):
    memory = resource_snapshot(torch_module, selection)
    total = memory.get("total_bytes")
    free = memory.get("free_bytes")
    hint = "缩小批次或输入尺寸，检查其他 GPU 进程，并确认模型/批次是否不必要地复制；实际可用内存会随运行变化。"
    if total is not None and free is not None:
        hint += f" 最近测得可用 GPU 内存 {free} / {total} 字节；这不是后续步骤的保证。"
    return UserError(
        "TRAINING_OUT_OF_MEMORY",
        f"训练在 {phase} 阶段耗尽设备 {selection.label} 的内存：{type(error).__name__}: {error}",
        hint,
        {"device": selection.label, "phase": phase, "resources": memory},
    )
