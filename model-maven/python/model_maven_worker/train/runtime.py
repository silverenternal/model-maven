"""Explicit dependencies shared by independent training engines."""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TrainingRuntime:
    torch: Any
    entrypoint: Callable[[str], Any]
    tree: Callable[[Any], Any]
    step_context: Callable[..., Any]
    scalar_value: Callable[..., float]
    scalar_map: Callable[..., dict]
    add_finite_total: Callable[..., None]
    move_batch: Callable[..., Any] | None = None
