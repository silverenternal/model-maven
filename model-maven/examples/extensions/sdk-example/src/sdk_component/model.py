"""A plain model factory that does not require an SDK base class or manifest."""

from torch import nn


def build_model(input_features: int = 4, output_features: int = 2) -> nn.Module:
    return nn.Sequential(nn.Linear(input_features, 8), nn.ReLU(), nn.Linear(8, output_features))
