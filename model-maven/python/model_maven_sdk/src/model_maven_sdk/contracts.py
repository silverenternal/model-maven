"""Public callable shapes for the currently supported worker entry points.

These are typing protocols only. They add no runtime base class and do not
claim that Model-Maven independently verifies user-owned algorithms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, TypedDict


class ModelFactory(Protocol):
    def __call__(self, **parameters: Any) -> Any: ...


class BatchFactory(Protocol):
    def __call__(self, **parameters: Any) -> Any: ...


class DatasetFactory(Protocol):
    def __call__(self, **parameters: Any) -> Any: ...


class Transform(Protocol):
    def __call__(self, sample: Any) -> Any: ...


class LossFunction(Protocol):
    def __call__(self, outputs: Any, batch: Any, **parameters: Any) -> Any: ...


class StepContext(Protocol):
    step: int
    total_steps: int

    def log(self, name: str, value: Any) -> None: ...

    def report_sample_ids(
        self, sample_ids: list[str] | tuple[str, ...], stable: bool = True
    ) -> None: ...


class ManagedStep(Protocol):
    def __call__(
        self, model: Any, batch: Any, context: StepContext, **parameters: Any
    ) -> Mapping[str, Any]: ...


class ExternalLoopContext(Protocol):
    """The narrow view exposed to a user-owned complete training loop."""

    total_steps: int
    completed_steps: int
    remaining_steps: int
    resume_state: Any
    resume_state_status: str
    stop_requested: bool
    timeout_requested: bool
    remaining_time_seconds: float | None

    @property
    def artifact_dir(self) -> Path: ...

    def log(self, name: str, value: Any) -> None: ...

    def report_step(
        self, loss: Any, metrics: Mapping[str, Any] | None = None
    ) -> bool: ...

    def report_sample_ids(
        self, sample_ids: list[str] | tuple[str, ...], stable: bool = True
    ) -> None: ...

    def register_artifact(
        self,
        artifact_id: str,
        files: list[str] | tuple[str, ...],
        *,
        owner: str = "external_loop",
        references: tuple[str, ...] = (),
        complete: bool = True,
        kind: str = "checkpoint",
    ) -> None: ...


class ExternalTrainingLoop(Protocol):
    def __call__(
        self, model: Any, context: ExternalLoopContext, **parameters: Any
    ) -> Any: ...


class EvaluatorContext(Protocol):
    """Finite result and optional sample-preview reporting for evaluators."""

    split: str
    max_samples: int | None
    max_preview_samples: int

    def log_metric(self, name: str, value: Any) -> None: ...

    def report_samples(self, count: int) -> None: ...

    def report_prediction(
        self,
        sample_id: str,
        prediction: Any,
        target: Any | None = None,
    ) -> None: ...


class Evaluator(Protocol):
    def __call__(
        self,
        model: Any,
        data: Any,
        context: EvaluatorContext,
        **parameters: Any,
    ) -> Mapping[str, Any]: ...


class StatefulCallable(Protocol):
    """Optional callable attributes used for explicit loop-state restoration."""

    def state_dict(self) -> Any: ...

    def load_state_dict(self, state: Any) -> None: ...


class VersionedStatefulCallable(StatefulCallable, Protocol):
    """Callable attributes bound by ``model_maven_sdk.bind_state_format``."""

    MODEL_MAVEN_STATE_FORMAT: str
    MODEL_MAVEN_STATE_VERSION: str
    MODEL_MAVEN_STATE_SCHEMA: Mapping[str, Any]
    MODEL_MAVEN_STATE_MIGRATOR: StateMigrator | None


class StateMigrator(Protocol):
    """User-owned pure conversion between two versions of one state format."""

    def __call__(
        self,
        state: Mapping[str, Any],
        *,
        from_version: str,
        to_version: str,
    ) -> Mapping[str, Any]: ...


class LayerEditOperation(TypedDict):
    """JSON operation persisted in a Layer Edit variant history."""

    action: str
    provider: str
    payload: dict[str, Any]


class LayerEditProviderV1(Protocol):
    """Compatibility protocol for a single callable Provider."""

    MODEL_MAVEN_LAYER_EDIT_API: int

    def __call__(
        self, model: Any, operation: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class LayerEditProvider(Protocol):
    """Lifecycle protocol for a replayable Layer Edit Provider API v2."""

    MODEL_MAVEN_LAYER_EDIT_API: int

    def describe(
        self, model: Any, operation: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def can_edit(
        self, model: Any, operation: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def plan(
        self, model: Any, operation: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def candidate(
        self, model: Any, plan: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def persist(self, plan: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def rollback(self, model: Any, plan: Mapping[str, Any]) -> Mapping[str, Any]: ...
