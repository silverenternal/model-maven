"""A plain batch factory using the native callable entry-point convention."""

import torch


def example_batch(batch_size: int = 2, input_features: int = 4):
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(batch_size, input_features, generator=generator)
    labels = torch.zeros(batch_size, dtype=torch.long)
    return {"args": [inputs], "kwargs": {}, "loss_inputs": {"labels": labels}}
