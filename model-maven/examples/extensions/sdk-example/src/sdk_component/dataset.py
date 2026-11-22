"""Plain Dataset factory for use with the native project data adapter."""

import torch
from torch.utils.data import TensorDataset


def build_dataset(samples: int = 8, input_features: int = 4, classes: int = 2):
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(samples, input_features, generator=generator)
    labels = torch.arange(samples) % classes
    return TensorDataset(values, labels)
