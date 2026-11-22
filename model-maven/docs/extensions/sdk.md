# Model-Maven Python extension SDK

`model-maven-sdk` is an optional, dependency-free Python package for authoring component metadata and typing the callable shapes that Model-Maven currently accepts. A component can still be a plain Python callable with a hand-written static manifest; installing the SDK is not a prerequisite for model use or one-off probing.

## Compatibility layers

These versions are intentionally separate:

| Version | Meaning |
| --- | --- |
| `model-maven-sdk` distribution version | Authoring helper release; currently `0.1.0`. |
| `schema_version` | Static component manifest structure; currently `1`. |
| `extension_api_version` | Component/Provider interface; currently `1`. |
| Layer Edit Provider API | v1 callable remains supported; v2 exposes the replayable describe/can_edit/plan/candidate/persist/rollback lifecycle. |
| Worker protocol | Negotiated between the Rust CLI and Python worker, independently of a component manifest. |
| State format version | `StateFormat(format_id, version, schema, migrators=...)` declares extension-owned state identity and exact upgrade routes; it is not inferred from SDK or distribution version. |

The current manifest validator does not accept a worker-protocol range field. The SDK does not invent one; protocol compatibility comes from the worker handshake and required operation capabilities.

The current worker accepts component manifest schema/API version `1` and rejects unknown versions and unknown top-level fields. It does not migrate a v0 or future manifest automatically. SDK `0.1.x` is an early authoring API; projects needing a different helper version should pin it explicitly. Since the SDK is optional, older projects using plain callables are not made dependent on an SDK upgrade.

## Minimal component

```python
from model_maven_sdk import Capability, ComponentManifest, object_schema, parameter

manifest = ComponentManifest(
    component_id="acme/scaled-identity",
    component_version="0.1.0",
    kind="layer",
    entrypoint="acme_layers.scale:make_layer",
    description="Scale values by a configured multiplier.",
    params_schema=object_schema({
        "scale": parameter(
            "number",
            description="Multiplier applied in forward.",
            default=1.0,
        ),
    }),
    capabilities={"construct": Capability("supported")},
)
```

Before building the wheel, call `write_static_manifest(package_dir, entry_point_name, manifest)`. It writes the discovery sidecar to `package_dir/__model_maven__/<entry_point_name>.json`, refuses to overwrite an existing file by default, and does not import the entry point. Include that JSON file as package data and register the same entry-point name under `model_maven.components` in `pyproject.toml`.

`parameter()` and `object_schema()` support the worker's documented JSON schema subset: object, array, string, integer, number, boolean, enum, numeric bounds, nested properties, required fields, defaults, and array items. Arbitrary JSON Schema keywords are rejected locally so a manifest does not silently rely on unsupported UI or validation behavior.

## Plain Python callable contracts

- Model, Dataset, and batch factories accept explicitly supplied keyword parameters; a forward batch can return a native value or an `{args, kwargs}` envelope, with optional user-owned `loss_inputs` for training.
- A managed step accepts `(model, batch, context, **parameters)` and returns a mapping containing a differentiable scalar `loss`, with optional `losses`, `metrics`, and `outputs`.
- An external loop accepts `(model, context, **parameters)`. `context.report_step(loss, metrics)` reports one completed user-owned update and returns false when either its step budget or cooperative deadline is reached. `context.remaining_time_seconds` and `context.timeout_requested` expose the runner's fixed deadline; check them before starting a step and honor the false return. The deadline is anchored when the worker starts, so model import/construction time consumes the same budget. This is cooperative: it cannot interrupt a blocked Python/CUDA call or undo an update already in progress. The CLI watchdog remains the process-level limit and, for formal training, requests a checkpoint at a reported boundary before force termination. A trainer that never reports a boundary cannot checkpoint in time, and processes that escape the worker process group remain outside cleanup guarantees.
- A registered `trainer` factory can be checked with `component test` when its manifest provides a `conformance_fixture`. The fixture may return `events`, `stop`, and/or `timeout` cases. Event cases contain `steps`, `expected_mean_loss`, and `expected_metrics`; stop cases contain a bounded `steps` budget; timeout cases contain `steps` and `timeout_seconds` (1–5). A declared `timeout: supported` or `conditional` requires a timeout fixture. The checker runs the real external-loop engine, validates that the trainer stops at a finite step boundary, and rejects a trainer that continues reporting updates after the deadline boundary. A passing fixture covers only its finite workload; it does not establish a wall-clock SLA for arbitrary steps or third-party code.
- An evaluator accepts `(model, data, context, **parameters)` and returns finite named `metrics`. The CLI can request a bounded sample preview with `m-model evaluate --preview-samples N` (1–32); the evaluator reports only records it owns through `context.report_prediction(sample_id, prediction, target=None)`. IDs must be unique within the report, values must be JSON-safe and finite, each record is limited to 64 KiB, and the complete preview is capped by the requested count. Tensor values must be explicitly converted to bounded JSON values (for example, a short list). `context.report_samples(count)` remains the source of the actual evaluated sample count; reporting previews does not imply a count or reinterpret custom artifacts.
- A Layer Edit Provider v1 is a callable accepting `(model, operation)` and returning a JSON-serializable summary. Provider v2 is an object exposing `describe`, `can_edit`, `plan`, `candidate`, `persist`, and `rollback`; it stores its JSON plan in the variant and must reproduce its persisted description on replay. Both declare `MODEL_MAVEN_LAYER_EDIT_API` in their module and remain subject to manifest, payload, fixture, and project-specific checks. Keep `describe`, `can_edit`, `plan`, and `persist` free of external side effects. `rollback` can compensate Provider-managed model changes; it cannot undo arbitrary file, network, or process effects.

The SDK protocols are typing aids; they do not impose inheritance. Context ports are minimal views of the currently implemented callables, not a promise that every context operation is available for every run type. A stateful custom training callable can bind an explicit JSON-compatible format and exact migration functions. The worker validates that state before saving and after loading, then invokes only the declared migration route before calling `load_state_dict`.

An extension can attach an independent state descriptor to its static manifest:

```python
from model_maven_sdk import ComponentManifest, StateFormat

manifest = ComponentManifest(
    component_id="acme/optimizer-state",
    component_version="3.0.0",
    kind="trainer",
    entrypoint="acme_training:make_trainer",
    state_schema=StateFormat(
        format_id="acme.adamw-state",
        version="2.1.0",
        schema={
            "type": "object",
            "properties": {
                "step": {"type": "integer", "description": "Optimizer step"},
                "moments": {"type": "object", "description": "Serialized moments"},
            },
            "required": ["step", "moments"],
            "additionalProperties": False,
        },
    ),
)
```

For a callable that exposes `state_dict()` and `load_state_dict(state)`, bind the same format to the function object:

```python
from model_maven_sdk import StateFormat, bind_state_format, parameter

def migrate_1_to_2(state, *, from_version, to_version):
    return {"step": state["step"], "revision": 2}

bind_state_format(
    train_loop,
    StateFormat(
        "acme.train-loop-state",
        "2.0.0",
        {
            "type": "object",
            "properties": {
                "step": parameter("integer", description="Completed step count"),
                "revision": parameter("integer", description="State schema revision"),
            },
            "required": ["step", "revision"],
            "additionalProperties": False,
        },
        migrators={"1.0.0->2.0.0": "acme_training.loop:migrate_1_to_2"},
    ),
    migrators={"1.0.0->2.0.0": migrate_1_to_2},
)
```

State schemas use the worker's JSON-compatible parameter Schema subset; tensors, arbitrary Python objects, and framework-specific values need an author-owned JSON representation. Migration is allowed only when a checkpoint contains the old format identity, the current callable declares the exact old-to-current route, and the callable's owning distribution is unambiguously identified. The worker also requires the Python ABI, PyTorch build, all other installed distributions, and all model/data/other training source files to match. Only changes within the single state callable's owning distribution are eligible. Unrelated source or environment changes, ambiguous distribution ownership, multiple migrated state callables, missing inventory, and mixed or chained format upgrades fail closed. There is no implicit network access, package installation, or checkpoint rewrite during validation; the next successful checkpoint records the current format.

## Package layout and verification

See [the independently buildable example](../../examples/extensions/sdk-example/README.md). It separates static component metadata from direct model, dataset, batch, loss, and training callables and demonstrates a local SDK install followed by component discovery. A successful SDK build does not prove extension conformance or algorithm correctness; those remain separate `component test` and end-to-end acceptance responsibilities.
