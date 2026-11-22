"""A tiny custom Layer factory; importable by Model-Maven after discovery."""

from torch import nn


class ScaledIdentity(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, values):
        return values * self.scale


def make_scaled_identity(scale: float = 1.0) -> ScaledIdentity:
    return ScaledIdentity(scale=scale)
