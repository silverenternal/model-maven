"""Small public authoring layer for Model-Maven extensions.

The SDK is optional. Plain Python callables and hand-written manifests remain
valid extension entry points.
"""

from .contracts import (
    BatchFactory,
    DatasetFactory,
    Evaluator,
    EvaluatorContext,
    ExternalLoopContext,
    ExternalTrainingLoop,
    LayerEditOperation,
    LayerEditProvider,
    LayerEditProviderV1,
    LossFunction,
    ManagedStep,
    ModelFactory,
    StatefulCallable,
    StateMigrator,
    StepContext,
    Transform,
    VersionedStatefulCallable,
)
from .manifest import (
    COMPONENT_SCHEMA_VERSION,
    EXTENSION_API_VERSION,
    SDK_VERSION,
    Capability,
    ComponentManifest,
    StateFormat,
    bind_state_format,
    object_schema,
    parameter,
    write_static_manifest,
)

__all__ = [
    "BatchFactory",
    "Capability",
    "COMPONENT_SCHEMA_VERSION",
    "ComponentManifest",
    "DatasetFactory",
    "Evaluator",
    "EvaluatorContext",
    "EXTENSION_API_VERSION",
    "ExternalLoopContext",
    "ExternalTrainingLoop",
    "LayerEditOperation",
    "LayerEditProvider",
    "LayerEditProviderV1",
    "LossFunction",
    "ManagedStep",
    "ModelFactory",
    "SDK_VERSION",
    "StatefulCallable",
    "StateFormat",
    "bind_state_format",
    "StateMigrator",
    "StepContext",
    "Transform",
    "VersionedStatefulCallable",
    "object_schema",
    "parameter",
    "write_static_manifest",
]
