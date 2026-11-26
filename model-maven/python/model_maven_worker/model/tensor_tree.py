"""Convert nested Python and tensor values into bounded JSON metadata."""

import math
from itertools import islice


def tree(torch_module, value, depth=0):
    if depth >= 6:
        return {"kind": "truncated"}
    if isinstance(value, torch_module.Tensor):
        if value.layout == torch_module.strided:
            if value.is_contiguous(memory_format=torch_module.channels_last_3d):
                memory_format = "channels_last_3d"
            elif value.is_contiguous(memory_format=torch_module.channels_last):
                memory_format = "channels_last"
            elif value.is_contiguous():
                memory_format = "contiguous"
            else:
                memory_format = "non_contiguous"
        else:
            memory_format = None
        return {
            "kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "layout": str(value.layout),
            "memory_format": memory_format, "requires_grad": value.requires_grad,
        }
    if isinstance(value, dict):
        count = len(value)
        items = list(islice(value.items(), 32))
        observed_keys_all_strings = all(type(key) is str for key, _ in items)
        result = {str(key): tree(torch_module, item, depth + 1) for key, item in items}
        if count > 32:
            result["<truncated>"] = count - 32
        return {"kind": "dict", "items": result,
                "keys_all_strings": False if not observed_keys_all_strings else (True if count <= 32 else None),
                "observed_keys_all_strings": observed_keys_all_strings,
                "items_truncated": count > 32}
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [tree(torch_module, item, depth + 1) for item in value[:32]],
            "truncated": max(0, len(value) - 32),
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value[:200] if isinstance(value, str) else value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return {"kind": "opaque", "type": type(value).__name__}

