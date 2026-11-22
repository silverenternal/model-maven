"""A small Layer factory exposed through a standard Python entry point."""

from torch import nn


def make_silu(inplace: bool = False) -> nn.SiLU:
    """Build a SiLU Layer with an explicit, schema-validated option."""
    return nn.SiLU(inplace=inplace)
