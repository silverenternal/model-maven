"""Compatibility import for forward input identity helpers.

The implementation lives in the shared tensor_inputs domain because both the
model probe and data-facing tooling use this bounded value codec.
"""

from ..tensor_inputs.content_identity import (
    ALGORITHM,
    MAX_BYTES,
    MAX_DEPTH,
    MAX_ITEMS,
    fingerprint,
)

__all__ = ["ALGORITHM", "MAX_BYTES", "MAX_DEPTH", "MAX_ITEMS", "fingerprint"]
