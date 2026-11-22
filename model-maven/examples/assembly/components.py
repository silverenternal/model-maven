"""Example component factory for a generated root-model assembly adapter."""

from torch import nn


def build_encoder(width: int = 4) -> nn.Module:
    return nn.Sequential(nn.Linear(width, width), nn.GELU())
