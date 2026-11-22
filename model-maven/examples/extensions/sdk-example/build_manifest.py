"""Generate the static discovery sidecar before building this wheel."""

from pathlib import Path

from model_maven_sdk import Capability, ComponentManifest, object_schema, parameter, write_static_manifest


PACKAGE = Path(__file__).parent / "src" / "sdk_component"

manifest = ComponentManifest(
    component_id="example/scaled-identity",
    component_version="0.1.0",
    kind="layer",
    entrypoint="sdk_component.layer:make_scaled_identity",
    description="Multiply each input value by an editable scale factor.",
    params_schema=object_schema(
        {"scale": parameter("number", description="Multiplier applied in forward.", default=1.0)},
        additional_properties=False,
    ),
    capabilities={"construct": Capability("supported")},
)

print(write_static_manifest(PACKAGE, "scaled-identity", manifest, overwrite=True))
