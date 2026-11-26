"""User-owned components for the from-zero, two-vector walkthrough.

Copy this file into a new project's root as ``dual_input_components.py``.
The widths are chosen by the project scaffold and passed as factory
parameters; this file does not select a model, dataset, Python, or device.
"""

import torch
from torch import nn


def vector_encoder(width: int) -> nn.Module:
    """Encode one dense vector without changing its feature width."""
    if width < 1:
        raise ValueError("width must be positive")
    return nn.Sequential(nn.Linear(width, width), nn.GELU())


class GatedConcat(nn.Module):
    """Scale each encoded vector independently, then concatenate features."""

    def __init__(self, inputs: int = 2):
        super().__init__()
        if inputs < 1:
            raise ValueError("inputs must be positive")
        self.scales = nn.Parameter(torch.ones(inputs))

    def forward(self, values):
        if len(values) != self.scales.numel():
            raise ValueError(
                f"expected {self.scales.numel()} encoded inputs, got {len(values)}"
            )
        if not values:
            raise ValueError("at least one encoded input is required")
        for index, value in enumerate(values):
            if value.ndim != 2:
                raise ValueError(f"encoded input {index} must have shape [batch, features]")
        batch_size = values[0].shape[0]
        for index, value in enumerate(values):
            if value.shape[0] != batch_size:
                raise ValueError("encoded inputs must have the same batch size")
        return torch.cat(
            [value * self.scales[index] for index, value in enumerate(values)],
            dim=1,
        )


class PlainConcat(nn.Module):
    """Concatenate the same encoded values without learned per-input scales."""

    def __init__(self, inputs: int = 2):
        super().__init__()
        if inputs < 1:
            raise ValueError("inputs must be positive")
        self.inputs = inputs

    def forward(self, values):
        if len(values) != self.inputs:
            raise ValueError(f"expected {self.inputs} encoded inputs, got {len(values)}")
        if not values:
            raise ValueError("at least one encoded input is required")
        if any(value.ndim != 2 for value in values):
            raise ValueError("encoded inputs must have shape [batch, features]")
        if any(value.shape[0] != values[0].shape[0] for value in values[1:]):
            raise ValueError("encoded inputs must have the same batch size")
        return torch.cat(values, dim=1)


def gated_concat(inputs: int = 2) -> nn.Module:
    """Factory used by the scaffold's ``fusion`` injection seam."""
    return GatedConcat(inputs=inputs)


def plain_concat(inputs: int = 2) -> nn.Module:
    """Replacement fusion factory with the same feature-width contract."""
    return PlainConcat(inputs=inputs)
