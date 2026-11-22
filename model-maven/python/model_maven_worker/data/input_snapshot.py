"""Compatibility imports for the shared bound-forward-input snapshot codec."""

from ..tensor_inputs.snapshot import (
    FORMAT,
    MAX_SNAPSHOT_BYTES,
    SnapshotError,
    encode,
    read_file,
    restore,
    write_new,
)

__all__ = [
    "FORMAT",
    "MAX_SNAPSHOT_BYTES",
    "SnapshotError",
    "encode",
    "read_file",
    "restore",
    "write_new",
]
