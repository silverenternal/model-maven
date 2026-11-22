"""Provider example that appends an activation to a model's encoder."""

from collections import OrderedDict

import torch
from torch import nn
from model_maven_silu_helper import validate_layer_name


MODEL_MAVEN_LAYER_EDIT_API = 1


def add_silu(model: nn.Module, operation: dict) -> dict:
    """Append a named SiLU module to ``model.encoder``."""
    name = validate_layer_name(operation["payload"]["name"])
    if not isinstance(model.encoder, nn.Sequential):
        raise TypeError("model.encoder must be nn.Sequential")

    modules = OrderedDict(model.encoder.named_children())
    if name in modules:
        raise ValueError(f"encoder already has a layer named {name!r}")
    modules[name] = nn.SiLU()
    model.encoder = nn.Sequential(modules)
    return {"added": f"model.encoder.{name}", "type": "SiLU"}


def provider_fixture() -> dict:
    """Return a fresh model and explicit expectations for the sample payload."""
    model = ExampleModel()
    return {
        "model": model,
        "expected_result": {"added": "model.encoder.silu", "type": "SiLU"},
        "expected_modules": {"encoder.silu": "SiLU"},
    }


class ExampleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            OrderedDict([("projection", nn.Linear(8, 8)), ("relu", nn.ReLU())])
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)
