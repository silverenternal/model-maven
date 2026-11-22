"""Dependency used by the independent SiLU Provider acceptance example."""


def validate_layer_name(name: object) -> str:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("name must be a valid Python identifier")
    return name
