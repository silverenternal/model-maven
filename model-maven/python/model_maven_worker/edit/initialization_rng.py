"""Explicit, bounded random-generator scopes for Layer construction."""

from contextlib import ExitStack, contextmanager
import re

from ..protocol import UserError


_DEVICE = re.compile(r"([a-z][a-z0-9_]*):([0-9]+)\Z")


def validate_rng_devices(value):
    """Validate declarative device IDs without importing or probing a backend."""
    if not isinstance(value, list) or len(value) > 16:
        raise UserError("LAYER_RNG_DEVICES_INVALID", "Layer 随机设备必须是最多 16 项的设备数组。")
    seen = set()
    backend = None
    result = []
    for value in value:
        match = _DEVICE.fullmatch(value) if isinstance(value, str) and len(value) <= 80 else None
        if not match or match.group(1) in {"cpu", "meta"}:
            raise UserError("LAYER_RNG_DEVICES_INVALID", "设备必须包含明确的非 CPU 后端和索引，例如 cuda:0。")
        kind, index_text = match.groups()
        index = int(index_text)
        if index > 65535 or index_text != str(index):
            raise UserError("LAYER_RNG_DEVICES_INVALID", "设备索引无效。")
        if backend is not None and backend != kind:
            raise UserError("LAYER_RNG_DEVICES_INVALID", "一次 Layer 初始化只能选择同一后端的设备。")
        if index in seen:
            raise UserError("LAYER_RNG_DEVICES_INVALID", "Layer 随机设备不能重复。")
        backend = kind
        seen.add(index)
        result.append(index)
    return backend, result


@contextmanager
def layer_rng_scope(torch, seed, devices):
    """Fork CPU and only explicitly selected accelerator generators."""
    backend, indexes = validate_rng_devices(devices)
    if not indexes:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            yield
        return
    device_module = getattr(torch, backend, None)
    if device_module is None:
        raise UserError("LAYER_RNG_BACKEND_UNAVAILABLE", f"所选 PyTorch 不支持随机设备后端 {backend!r}。")
    with ExitStack() as stack:
        try:
            stack.enter_context(torch.random.fork_rng(devices=indexes, device_type=backend))
            generators = getattr(device_module, "default_generators", None)
            if generators is None:
                raise UserError("LAYER_RNG_BACKEND_UNAVAILABLE", f"后端 {backend!r} 未公开默认设备生成器。")
            torch.random.default_generator.manual_seed(seed)
            for index in indexes:
                generators[index].manual_seed(seed)
        except UserError:
            raise
        except Exception as error:
            raise UserError("LAYER_RNG_DEVICE_FAILED", f"无法使用所选随机设备：{type(error).__name__}: {error}",
                            "确认所选解释器、设备索引及后端 API 可用；不会自动改用其他设备。") from error
        yield
