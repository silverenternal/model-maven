"""Generated Model-Maven assembly adapter."""
# Generated from an explicit Model-Maven assembly recipe (schema v1).
# This adapter does not vendor user modules, extension packages, or runtime dependencies.
from importlib import import_module


def _resolve(entrypoint):
    module_name, attribute = entrypoint.split(":", 1)
    value = import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def build_model():
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        return _build_model()


def _build_model():
    # Description: "A user-owned classifier root with a separately declared encoder seam."
    # Declared root-model seam: encoder
    _component_0 = _resolve('components:build_encoder')(**{'width': 4})
    return _resolve('root_model:build_model')(**{'classes': 3}, encoder=_component_0)


if __name__ == "__main__":
    model = build_model()
    print(type(model).__qualname__)
