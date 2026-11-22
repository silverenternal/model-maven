"""Bounded data-only snapshots of supported bound forward inputs.

This codec deliberately does not use pickle or import classes from a snapshot.
It only covers the content-identity subset and refuses cyclic containers.
Callers are responsible for private storage, lifecycle, and explicit identity
checks against the diagnosis they intend to replay.
"""

import base64
import binascii
import json
import math
import os
import stat
import struct

from .content_identity import MAX_BYTES, MAX_DEPTH, MAX_ITEMS, fingerprint

FORMAT = "model-maven-forward-input-snapshot-v1"
MAX_SNAPSHOT_BYTES = 96 * 1024 * 1024
_DTYPES = {
    "bool", "uint8", "int8", "int16", "int32", "int64", "float16",
    "float32", "float64", "bfloat16", "complex64", "complex128",
}


class SnapshotError(ValueError):
    """The input is unsupported or the snapshot is invalid."""


def write_new(path, payload):
    """Create a private snapshot without overwriting or following a final symlink."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(payload)
            stream.flush()
    except BaseException:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def read_file(path):
    """Read only a bounded regular file, rejecting final symlinks when supported."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SNAPSHOT_BYTES:
            raise SnapshotError("snapshot file is not a bounded regular file")
        payload = stream.read(MAX_SNAPSHOT_BYTES + 1)
        if len(payload) > MAX_SNAPSHOT_BYTES:
            raise SnapshotError("snapshot exceeds byte limit")
        return payload


def encode(args, kwargs, torch):
    identity = fingerprint(args, kwargs, torch)
    if identity.get("status") != "available":
        raise SnapshotError("input content cannot be completely fingerprinted")
    seen, active = {}, set()
    count = 0

    def visit(value, depth):
        nonlocal count
        count += 1
        if count > MAX_ITEMS or depth > MAX_DEPTH:
            raise SnapshotError("snapshot structure limit exceeded")
        kind = type(value)
        if kind is type(None):
            return ["none"]
        if kind is bool:
            return ["bool", value]
        if kind is int:
            return ["int", str(value)]
        if kind is float:
            return ["float", struct.pack(">d", value).hex()]
        if kind is str:
            return ["str", value]
        if kind not in (tuple, list, dict, torch.Tensor):
            raise SnapshotError("unsupported input type")
        key = id(value)
        if key in active:
            raise SnapshotError("cyclic input containers are unsupported")
        if key in seen:
            return ["ref", seen[key]]
        index = len(seen)
        seen[key] = index
        active.add(key)
        try:
            if kind is torch.Tensor:
                if (value._base is not None or value.grad_fn is not None or value.is_pinned()
                        or value.is_inference() or any(name is not None for name in getattr(value, "names", ()))
                        or value.untyped_storage().nbytes() != value.numel() * value.element_size()):
                    raise SnapshotError("tensor view or storage state is unsupported")
                dtype = str(value.dtype).removeprefix("torch.")
                if dtype not in _DTYPES:
                    raise SnapshotError("tensor dtype is unsupported")
                raw = value.detach().reshape(-1).view(torch.uint8).numpy().tobytes()
                return ["tensor", index, dtype, list(value.shape), list(value.stride()),
                        value.requires_grad, base64.b64encode(raw).decode("ascii")]
            if kind is dict:
                if any(type(key) is not str for key in value):
                    raise SnapshotError("dictionary keys must be strings")
                return ["dict", index, [[key, visit(item, depth + 1)] for key, item in value.items()]]
            return ["tuple" if kind is tuple else "list", index,
                    [visit(item, depth + 1) for item in value]]
        finally:
            active.remove(key)

    tree = visit((args, kwargs), 0)
    payload = json.dumps({"format": FORMAT, "digest": identity["digest"], "tree": tree},
                         ensure_ascii=True, separators=(",", ":")).encode("ascii")
    if len(payload) > MAX_SNAPSHOT_BYTES:
        raise SnapshotError("snapshot exceeds byte limit")
    # Refuse values that cannot be restored exactly on this runtime.
    restore(payload, torch, expected_digest=identity["digest"])
    return payload


def restore(payload, torch, *, expected_digest):
    if type(payload) is not bytes or len(payload) > MAX_SNAPSHOT_BYTES:
        raise SnapshotError("snapshot exceeds byte limit")
    if (type(expected_digest) is not str or len(expected_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_digest)):
        raise SnapshotError("expected input digest is invalid")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotError("duplicate snapshot key")
            result[key] = value
        return result

    try:
        document = json.loads(payload, object_pairs_hook=unique_pairs,
                              parse_constant=lambda _: (_ for _ in ()).throw(SnapshotError("invalid JSON constant")))
        if (type(document) is not dict or set(document) != {"format", "digest", "tree"}
                or document["format"] != FORMAT or document["digest"] != expected_digest):
            raise SnapshotError("snapshot identity mismatch")
        objects, count, total_bytes, next_index = {}, 0, 0, 0

        def visit(node, depth):
            nonlocal count, total_bytes, next_index
            count += 1
            if count > MAX_ITEMS or depth > MAX_DEPTH or type(node) is not list or not node:
                raise SnapshotError("snapshot structure limit exceeded")
            tag = node[0]
            if tag == "none" and len(node) == 1:
                return None
            if tag == "bool" and len(node) == 2 and type(node[1]) is bool:
                return node[1]
            if tag == "int" and len(node) == 2 and type(node[1]) is str:
                return int(node[1])
            if tag == "float" and len(node) == 2 and type(node[1]) is str and len(node[1]) == 16:
                return struct.unpack(">d", bytes.fromhex(node[1]))[0]
            if tag == "str" and len(node) == 2 and type(node[1]) is str:
                return node[1]
            if tag == "ref" and len(node) == 2 and type(node[1]) is int and node[1] in objects:
                return objects[node[1]]
            if (tag not in ("tensor", "dict", "tuple", "list") or len(node) < 3
                    or type(node[1]) is not int or node[1] != next_index):
                raise SnapshotError("invalid snapshot node")
            index = next_index
            next_index += 1
            if tag == "tensor":
                if len(node) != 7:
                    raise SnapshotError("invalid tensor")
                _, _, dtype_name, shape, stride, grad, encoded = node
                if (dtype_name not in _DTYPES or type(shape) is not list or type(stride) is not list
                        or len(shape) != len(stride) or len(shape) > MAX_DEPTH
                        or any(type(n) is not int or n < 0 or n > MAX_BYTES for n in shape + stride)
                        or type(grad) is not bool or type(encoded) is not str):
                    raise SnapshotError("invalid tensor metadata")
                dtype = getattr(torch, dtype_name)
                item_size = torch.empty((), dtype=dtype).element_size()
                size = math.prod(shape) * item_size
                storage_size = (1 + sum((n - 1) * s for n, s in zip(shape, stride))) * item_size if all(shape) else 0
                if size > MAX_BYTES or storage_size > MAX_BYTES or len(encoded) > ((MAX_BYTES + 2) // 3) * 4:
                    raise SnapshotError("tensor byte limit exceeded")
                raw = base64.b64decode(encoded, validate=True)
                total_bytes += len(raw)
                if len(raw) != size or total_bytes > MAX_BYTES:
                    raise SnapshotError("tensor payload size mismatch")
                value = torch.empty_strided(shape, stride, dtype=dtype)
                if raw:
                    value.copy_(torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape))
                value.requires_grad_(grad)
            else:
                if len(node) != 3 or type(node[2]) is not list:
                    raise SnapshotError("invalid container")
                if tag == "dict":
                    value = {}
                    for item in node[2]:
                        if (type(item) is not list or len(item) != 2 or type(item[0]) is not str
                                or item[0] in value):
                            raise SnapshotError("invalid dictionary")
                        value[item[0]] = visit(item[1], depth + 1)
                else:
                    items = [visit(item, depth + 1) for item in node[2]]
                    value = tuple(items) if tag == "tuple" else items
            objects[index] = value
            return value

        result = visit(document["tree"], 0)
        if type(result) is not tuple or len(result) != 2 or type(result[0]) is not tuple or type(result[1]) is not dict:
            raise SnapshotError("invalid bound arguments")
        identity = fingerprint(result[0], result[1], torch)
        if identity.get("status") != "available" or identity.get("digest") != expected_digest:
            raise SnapshotError("restored input identity mismatch")
        return result
    except SnapshotError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RuntimeError, UnicodeError, binascii.Error, RecursionError) as error:
        raise SnapshotError("invalid or unsupported snapshot") from error
