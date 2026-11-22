# Model-Maven SDK example

This package shows the SDK as an optional authoring aid. It contains a packaged Layer component with a static manifest, plus ordinary model, batch, and managed-step callables. The callable entry points do not inherit from an SDK base class.

## Build and install

From this directory, install the local SDK distribution, then build/install the example in the same Python environment used by Model-Maven:

```sh
python -m pip install ../../../python/model_maven_sdk
python build_manifest.py
python -m pip install .
m-model --python "$(python -c 'import sys; print(sys.executable)')" component discover --distribution model-maven-sdk-example
m-model component show example/scaled-identity
m-model component test example/scaled-identity --param scale=0.5
```

Choose a Python environment with a PyTorch version supported by the project before installing the example. Discovery reads `src/sdk_component/__model_maven__/scaled-identity.json` statically; it does not import the package while scanning entry points. The SDK distribution itself has no runtime dependencies.

## Use plain callables

The package's `sdk_component.model:build_model`, `sdk_component.batch:example_batch`, and `sdk_component.training:training_step` are regular Python functions. They can be selected by the normal Model-Maven model, batch, and step options; the optional SDK contributes typing protocols but does not wrap or execute them.

For example, the component manifest builder uses the public schema helpers and emits the same schema/API version fields consumed by component discovery. `component_version`, `extension_api_version`, the SDK package version, worker protocol negotiation, and any checkpoint state format have separate meanings. This current component manifest format does not accept a worker-protocol range field; protocol compatibility is reported by the Rust/Python worker handshake.

The example demonstrates component construction metadata only. It does not claim custom Layer edit-provider, persistence, conformance, state migration, or arbitrary training-loop support.
