import json
import tempfile
import unittest
from pathlib import Path

from model_maven_sdk import (
    Capability,
    ComponentManifest,
    object_schema,
    parameter,
    StateFormat,
    bind_state_format,
    write_static_manifest,
)


def upgrade_state(state, *, from_version, to_version):
    if (from_version, to_version) != ("1.0.0", "2.0.0"):
        raise ValueError("unsupported migration route")
    return {"counter": state["counter"], "revision": 2}


def valid_manifest(**changes):
    values = {
        "component_id": "example/scaler",
        "component_version": "1.2.3",
        "kind": "layer",
        "entrypoint": "example_layer:make_scaler",
        "params_schema": object_schema(
            {"scale": parameter("number", description="Scale factor", default=1.0)}
        ),
        "capabilities": {"construct": Capability("supported")},
    }
    values.update(changes)
    return ComponentManifest(**values)


class ManifestTests(unittest.TestCase):
    def test_manifest_uses_current_component_and_extension_versions(self):
        result = valid_manifest().to_dict()
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["extension_api_version"], 1)
        self.assertEqual(result["namespace"], "example")

    def test_manifest_rejects_invalid_version_kind_entrypoint_and_schema(self):
        invalid = [
            {"component_version": "01.2.3"},
            {"kind": "unknown"},
            {"entrypoint": "not-a-callable"},
            {"params_schema": parameter("number", description="wrong top-level")},
            {"capabilities": {"edit": Capability("conditional")}},
            {"dependencies": ({"component_id": "example/base", "version": "garbage"},)},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                valid_manifest(**changes).to_dict()

    def test_manifest_writer_creates_entrypoint_sidecar_without_importing_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            sidecar = write_static_manifest(package, "scaler", valid_manifest())
            self.assertEqual(sidecar.relative_to(package).as_posix(), "__model_maven__/scaler.json")
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), valid_manifest().to_dict())
            with self.assertRaises(FileExistsError):
                write_static_manifest(package, "scaler", valid_manifest())

    def test_schema_builders_reject_invalid_defaults_and_bounds(self):
        with self.assertRaises(ValueError):
            parameter("number", description="Scale factor", default=float("nan"))
        with self.assertRaises(ValueError):
            parameter("integer", description="Count", minimum=5, maximum=1)
        with self.assertRaises(ValueError):
            object_schema({"bad": parameter("string", description="Name")}, required=("absent",))

    def test_state_format_version_is_independent_and_serialized_in_metadata(self):
        manifest = valid_manifest(
            state_schema=StateFormat(
                format_id="example.optimizer-state",
                version="2.1.0",
                schema={
                    "type": "object",
                    "properties": {"step": parameter("integer", description="Step")},
                    "required": ["step"],
                    "additionalProperties": False,
                },
            )
        ).to_dict()
        self.assertEqual(
            manifest["state_schema"],
            {
                "format_id": "example.optimizer-state",
                "version": "2.1.0",
                "schema": {
                    "type": "object",
                    "properties": {"step": parameter("integer", description="Step")},
                    "required": ["step"],
                    "additionalProperties": False,
                },
            },
        )
        with self.assertRaises(ValueError):
            StateFormat("example.optimizer-state", "02.1.0", {}).to_dict()

    def test_state_format_migrator_is_explicit_and_bound_to_callable(self):
        state_format = StateFormat(
            "example.loop-state",
            "2.0.0",
            {
                "type": "object",
                "properties": {
                    "counter": parameter("integer", description="Counter"),
                    "revision": parameter("integer", description="Revision"),
                },
                "required": ["counter", "revision"],
                "additionalProperties": False,
            },
            migrators={
                "1.0.0->2.0.0": f"{upgrade_state.__module__}:upgrade_state"
            },
        )

        def run_loop():
            return None

        self.assertIs(
            bind_state_format(
                run_loop,
                state_format,
                migrators={"1.0.0->2.0.0": upgrade_state},
            ),
            run_loop,
        )
        self.assertEqual(run_loop.MODEL_MAVEN_STATE_FORMAT, "example.loop-state")
        self.assertEqual(run_loop.MODEL_MAVEN_STATE_VERSION, "2.0.0")
        self.assertIs(run_loop.MODEL_MAVEN_STATE_MIGRATORS["1.0.0->2.0.0"], upgrade_state)
        self.assertEqual(
            state_format.to_dict()["migrators"],
            {"1.0.0->2.0.0": f"{upgrade_state.__module__}:upgrade_state"},
        )
        with self.assertRaises(ValueError):
            bind_state_format(run_loop, state_format, migrators={})


if __name__ == "__main__":
    unittest.main()
