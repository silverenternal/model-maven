"""Optional user-owned components for the two-input starter walkthrough.

Copy this file into the new project. The widths are factory parameters so the
same components can be adapted to other vector inputs; no dataset is bundled.
"""

import torch
from torch import nn


def vector_encoder(width=4):
    """Transform a vector while preserving its final feature dimension."""
    return nn.Sequential(nn.Linear(width, width), nn.GELU())


class GatedConcat(nn.Module):
    """Learn one scale per input before concatenating encoded features."""

    def __init__(self, inputs=2):
        super().__init__()
        self.scales = nn.Parameter(torch.ones(inputs))

    def forward(self, values):
        if len(values) != self.scales.numel():
            raise ValueError("GatedConcat expects the configured number of inputs")
        return torch.cat(
            [value * self.scales[index] for index, value in enumerate(values)],
            dim=1,
        )


def gated_concat(inputs=2):
    return GatedConcat(inputs=inputs)
