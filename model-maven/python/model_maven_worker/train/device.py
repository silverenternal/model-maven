"""Explicit single-device placement, nested batch movement and resource reports."""

import re
import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..protocol import UserError


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
    """Resolve CPU, automatic preference, or one visible CUDA-compatible device."""
    if requested is None:
        requested = "cpu"
    if not isinstance(requested, str):
        raise UserError("DEVICE_INVALID", "设备必须是 cpu、auto、cuda 或 cuda:N。")
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
            requested = "cuda:0"
            selection = _resolve_explicit_gpu(torch_module, requested, requested_by="auto")
            return selection
        device = torch_module.device("cpu")
        return DeviceSelection(device, "cpu", "cpu", "CPU", "auto")
    if requested == "cpu":
        device = torch_module.device("cpu")
        return DeviceSelection(device, "cpu", "cpu", "CPU", "cpu")

    match = re.fullmatch(r"cuda(?::([0-9]+))?", requested)
    if not match:
        raise UserError(
            "DEVICE_INVALID",
            f"不支持设备标识 {requested!r}；首版接受 cpu、auto、cuda 或 cuda:N。",
            "使用 --device cpu、--device auto，或选择一个当前 Python 环境可见的 GPU，例如 --device cuda:0。",
        )
    return _resolve_explicit_gpu(torch_module, requested, requested_by=requested)


def _resolve_explicit_gpu(torch_module, requested, requested_by):
    match = re.fullmatch(r"cuda(?::([0-9]+))?", requested)
    if not match:
        raise UserError("DEVICE_INVALID", "首版 GPU 设备必须使用 cuda 或 cuda:N。")
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise UserError(
            "DEVICE_UNAVAILABLE",
            "请求了 GPU，但当前 Python/PyTorch 环境没有可用的 CUDA/ROCm 设备。",
            "检查当前解释器的 PyTorch 构建、GPU 驱动和设备可见性；训练未启动。",
            {"requested_device": requested, "visible_device_count": 0},
        )
    count = cuda.device_count()
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


def move_nested_batch(torch_module, value, device):
    """Move Tensor leaves in mappings and built-in sequences; retain other values."""
    if isinstance(value, torch_module.Tensor):
        return value.to(device)
    if isinstance(value, OrderedDict):
        return type(value)((key, move_nested_batch(torch_module, item, device)) for key, item in value.items())
    if isinstance(value, Mapping):
        moved = {key: move_nested_batch(torch_module, item, device) for key, item in value.items()}
        if type(value) is dict:
            return moved
        try:
            return type(value)(moved)
        except (TypeError, ValueError):
            return moved
    if isinstance(value, list):
        return [move_nested_batch(torch_module, item, device) for item in value]
    if isinstance(value, tuple):
        moved = tuple(move_nested_batch(torch_module, item, device) for item in value)
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
