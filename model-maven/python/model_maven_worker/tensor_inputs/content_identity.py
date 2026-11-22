"""Bounded, conservative content identity for model forward arguments.

The digest is not a data replay facility, model identity, or proof of runtime
equivalence. Unsupported inputs are never assigned a partial digest.
"""

import hashlib
import struct

ALGORITHM = "sha256-forward-input-v1"
MAX_BYTES = 64 * 1024 * 1024
MAX_ITEMS = 4096
MAX_DEPTH = 32


class _Unsupported(Exception):
    pass


def fingerprint(args, kwargs, torch_module):
    digest = hashlib.sha256()
    digest.update(b"model-maven-forward-input-v1\0")
    seen = {}
    storages = {}
    count = 0
    total_bytes = 0

    def field(tag, value=b""):
        nonlocal total_bytes
        total_bytes += len(value)
        if total_bytes > MAX_BYTES:
            raise _Unsupported("input exceeds content fingerprint byte limit")
        digest.update(tag)
        digest.update(struct.pack(">Q", len(value)))
        digest.update(value)

    def visit(value, depth):
        nonlocal count
        count += 1
        if count > MAX_ITEMS or depth > MAX_DEPTH:
            raise _Unsupported("input exceeds content fingerprint structure limit")
        kind = type(value)
        if kind is type(None):
            field(b"N")
        elif kind is bool:
            field(b"B", b"1" if value else b"0")
        elif kind is int:
            field(b"I", str(value).encode("ascii"))
        elif kind is float:
            field(b"F", struct.pack(">d", value))
        elif kind is str:
            field(b"S", value.encode("utf-8"))
        elif kind in (tuple, list, dict) or kind is torch_module.Tensor:
            identifier = id(value)
            if identifier in seen:
                field(b"R", struct.pack(">Q", seen[identifier]))
                return
            seen[identifier] = len(seen)
            if kind is torch_module.Tensor:
                if (
                    value.device.type != "cpu"
                    or value.layout != torch_module.strided
                    or value.is_quantized
                    or not value.is_contiguous()
                    or value.storage_offset() != 0
                    or value.is_conj()
                    or value.is_neg()
                ):
                    raise _Unsupported("tensor layout or device is unsupported")
                storage = value.untyped_storage()._cdata
                if storage in storages:
                    raise _Unsupported("distinct tensors share storage")
                storages[storage] = identifier
                size = value.numel() * value.element_size()
                if size > MAX_BYTES - total_bytes:
                    raise _Unsupported("input exceeds content fingerprint byte limit")
                field(b"T", str(value.dtype).encode("ascii"))
                field(b"G", b"1" if value.requires_grad else b"0")
                field(b"D", struct.pack(">Q", value.dim()))
                for dim, stride in zip(value.shape, value.stride()):
                    field(b"d", struct.pack(">q", dim))
                    field(b"s", struct.pack(">q", stride))
                field(
                    b"V",
                    value.detach()
                    .reshape(-1)
                    .view(torch_module.uint8)
                    .numpy()
                    .tobytes(),
                )
            elif kind is dict:
                field(b"M", struct.pack(">Q", len(value)))
                for key, item in value.items():
                    if type(key) is not str:
                        raise _Unsupported("dictionary keys must be strings")
                    visit(key, depth + 1)
                    visit(item, depth + 1)
            else:
                field(b"L" if kind is list else b"U", struct.pack(">Q", len(value)))
                for item in value:
                    visit(item, depth + 1)
        else:
            raise _Unsupported("input contains an unsupported value type")

    try:
        visit((args, kwargs), 0)
        return {
            "status": "available",
            "algorithm": ALGORITHM,
            "digest": digest.hexdigest(),
        }
    except Exception as error:
        # Do not leak user values or present partial coverage as a match.
        reason = (
            str(error)
            if isinstance(error, _Unsupported)
            else "input could not be fingerprinted"
        )
        return {"status": "unsupported", "algorithm": ALGORITHM, "reason": reason}
