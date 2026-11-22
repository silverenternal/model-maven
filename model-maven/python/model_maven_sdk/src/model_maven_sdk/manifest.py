"""Build and write the static sidecar manifest used by component discovery."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SDK_VERSION = "0.1.0"
COMPONENT_SCHEMA_VERSION = 1
EXTENSION_API_VERSION = 1

_KINDS = {
    "model",
    "layer",
    "dataset",
    "transform",
    "loss",
    "metric",
    "trainer",
    "layer-edit-provider",
}
_STATES = {"supported", "unsupported", "unknown", "conditional"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEGMENT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SEMVER_PART = re.compile(r"^(?:0|[1-9][0-9]*|[xX*])$")
_STATE_FORMAT_ID = re.compile(r"^[a-z][a-z0-9_.-]{1,127}$")
_SEMVER_SUFFIX = re.compile(
    r"^(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


@dataclass(frozen=True)
class Capability:
    """A capability claim with explicit limits for unsupported/conditional cases."""

    state: str
    reason: str | None = None
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any] | str:
        if not isinstance(self.state, str) or self.state not in _STATES:
            raise ValueError(f"unsupported capability state: {self.state!r}")
        if self.state == "supported" and self.reason is None and not self.requires:
            return self.state
        if self.state != "supported" and not self.reason:
            raise ValueError(f"capability state {self.state!r} requires a reason")
        if any(not isinstance(value, str) or not value.strip() for value in self.requires):
            raise ValueError("capability requires entries must be non-empty strings")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError("capability requires entries must be unique")
        result: dict[str, Any] = {"state": self.state}
        if self.reason is not None:
            result["reason"] = self.reason
        if self.requires:
            result["requires"] = list(self.requires)
        return result


@dataclass(frozen=True)
class ComponentManifest:
    """Static component metadata; it never imports or executes the component."""

    component_id: str
    component_version: str
    kind: str
    entrypoint: str
    description: str = ""
    params_schema: Mapping[str, Any] = field(
        default_factory=lambda: object_schema({})
    )
    capabilities: Mapping[str, Capability | str | Mapping[str, Any]] = field(
        default_factory=dict
    )
    dependencies: tuple[Mapping[str, Any], ...] = ()
    docs: str | Mapping[str, Any] | None = None
    port_contracts: Mapping[str, Any] | None = None
    renderer: Mapping[str, Any] | None = None
    state_schema: StateFormat | Mapping[str, Any] | None = None
    conformance_fixture: str | None = None

    def to_dict(self) -> dict[str, Any]:
        namespace = _validate_component_id(self.component_id)
        if not _SEMVER.fullmatch(self.component_version):
            raise ValueError("component_version must use MAJOR.MINOR.PATCH without leading zeroes")
        if not isinstance(self.kind, str) or self.kind not in _KINDS:
            raise ValueError(f"unsupported component kind: {self.kind!r}")
        _validate_entrypoint(self.entrypoint, "entrypoint")
        if not isinstance(self.description, str) or len(self.description) > 500:
            raise ValueError("description must be a string of at most 500 characters")
        if not isinstance(self.params_schema, Mapping) or self.params_schema.get("type") != "object":
            raise ValueError("params_schema must be a top-level object schema")
        validate_parameter_schema(self.params_schema)
        capabilities = {
            name: value.to_dict() if isinstance(value, Capability) else value
            for name, value in self.capabilities.items()
        }
        _validate_capabilities(capabilities)
        dependencies = _validate_dependencies(self.dependencies)

        manifest: dict[str, Any] = {
            "schema_version": COMPONENT_SCHEMA_VERSION,
            "namespace": namespace,
            "component_id": self.component_id,
            "component_version": self.component_version,
            "extension_api_version": EXTENSION_API_VERSION,
            "kind": self.kind,
            "entrypoint": self.entrypoint,
            "description": self.description,
            "params_schema": dict(self.params_schema),
            "capabilities": capabilities,
        }
        if dependencies:
            manifest["dependencies"] = dependencies
        if self.docs is not None:
            if not isinstance(self.docs, (str, Mapping)):
                raise ValueError("docs must be a string or object")
            manifest["docs"] = self.docs
        if self.port_contracts is not None:
            if not isinstance(self.port_contracts, Mapping):
                raise ValueError("port_contracts must be an object")
            manifest["port_contracts"] = dict(self.port_contracts)
        if self.renderer is not None:
            if not isinstance(self.renderer, Mapping):
                raise ValueError("renderer must be an object")
            manifest["renderer"] = dict(self.renderer)
        if self.state_schema is not None:
            if not isinstance(self.state_schema, (Mapping, StateFormat)):
                raise ValueError("state_schema must be an object")
            if isinstance(self.state_schema, StateFormat):
                manifest["state_schema"] = self.state_schema.to_dict()
            else:
                raw_state_schema = dict(self.state_schema)
                if (
                    set(raw_state_schema) - {"format_id", "version", "schema", "migrators"}
                    or not {"format_id", "version", "schema"}.issubset(raw_state_schema)
                ):
                    raise ValueError("state_schema must contain format_id, version, schema, and optional migrators")
                manifest["state_schema"] = StateFormat(**raw_state_schema).to_dict()
        if self.conformance_fixture is not None:
            _validate_entrypoint(self.conformance_fixture, "conformance_fixture")
            manifest["conformance_fixture"] = self.conformance_fixture
        # Fail at authoring time for data that cannot be represented as JSON.
        json.dumps(manifest, ensure_ascii=False, allow_nan=False)
        return manifest


@dataclass(frozen=True)
class StateFormat:
    """Opt-in, separately versioned descriptor for extension-owned state."""

    format_id: str
    version: str
    schema: Mapping[str, Any]
    migrators: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if not isinstance(self.format_id, str) or not _STATE_FORMAT_ID.fullmatch(self.format_id):
            raise ValueError("state format_id must be a stable lowercase identifier")
        if not isinstance(self.version, str) or not _SEMVER.fullmatch(self.version):
            raise ValueError("state format version must use MAJOR.MINOR.PATCH")
        if not isinstance(self.schema, Mapping):
            raise ValueError("state format schema must be an object")
        if self.schema.get("type") != "object":
            raise ValueError("state format schema must describe an object")
        validate_parameter_schema(self.schema)
        result = {
            "format_id": self.format_id,
            "version": self.version,
            "schema": dict(self.schema),
        }
        if not isinstance(self.migrators, Mapping):
            raise ValueError("state format migrators must be a mapping")
        routes = {}
        for route, entrypoint in self.migrators.items():
            if not isinstance(route, str) or route.count("->") != 1:
                raise ValueError("state migrator keys must be FROM_VERSION->TO_VERSION")
            from_version, to_version = route.split("->", 1)
            if not _SEMVER.fullmatch(from_version) or not _SEMVER.fullmatch(to_version):
                raise ValueError("state migrator versions must use MAJOR.MINOR.PATCH")
            if tuple(map(int, to_version.split("."))) <= tuple(map(int, from_version.split("."))):
                raise ValueError("state migrators may only upgrade to a newer version")
            _validate_entrypoint(entrypoint, f"state migrator {route}")
            routes[route] = entrypoint
        if len(routes) != len(self.migrators):
            raise ValueError("state migrator routes must be unique")
        if routes:
            result["migrators"] = routes
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result


def bind_state_format(
    function: Any,
    state_format: StateFormat | Mapping[str, Any],
    migrators: Mapping[str, Any] | None = None,
) -> Any:
    """Bind a JSON-safe state format and optional upgrader to a stateful callable.

    The worker snapshots this descriptor with ``state_dict()`` output. A
    migrator runs only when an explicit checkpoint carries the same format ID
    at an older version; the worker validates its result before calling
    ``load_state_dict()``. ``StateFormat.migrators`` declares exact version
    routes and entry points; the supplied callable mapping must match it.
    """
    if not callable(function):
        raise ValueError("state format can only be bound to a callable")
    if isinstance(state_format, StateFormat):
        descriptor = state_format.to_dict()
    elif isinstance(state_format, Mapping):
        raw = dict(state_format)
        if set(raw) - {"format_id", "version", "schema", "migrators"} or not {"format_id", "version", "schema"}.issubset(raw):
            raise ValueError("state format must contain format_id, version, schema, and optional migrators")
        descriptor = StateFormat(**raw).to_dict()
    else:
        raise ValueError("state_format must be a StateFormat or descriptor mapping")
    declared = descriptor.get("migrators", {})
    supplied = dict(migrators or {})
    if set(supplied) != set(declared):
        raise ValueError("state migrator callables must match the StateFormat manifest routes exactly")
    migration_manifest = []
    for route, migrator in supplied.items():
        if not callable(migrator):
            raise ValueError(f"state migrator {route} must be callable")
        entrypoint = f"{migrator.__module__}:{migrator.__qualname__}"
        if entrypoint != declared[route]:
            raise ValueError(f"state migrator {route} callable does not match its manifest entrypoint")
        from_version, to_version = route.split("->", 1)
        migration_manifest.append({
            "format_id": descriptor["format_id"],
            "from_version": from_version,
            "to_version": to_version,
            "entrypoint": entrypoint,
        })
    try:
        setattr(function, "MODEL_MAVEN_STATE_FORMAT", descriptor["format_id"])
        setattr(function, "MODEL_MAVEN_STATE_VERSION", descriptor["version"])
        setattr(function, "MODEL_MAVEN_STATE_SCHEMA", descriptor["schema"])
        setattr(function, "MODEL_MAVEN_STATE_MIGRATION_MANIFEST", migration_manifest)
        setattr(function, "MODEL_MAVEN_STATE_MIGRATORS", supplied)
    except (AttributeError, TypeError) as error:
        raise ValueError("callable does not allow binding state format metadata") from error
    return function


def parameter(
    kind: str,
    *,
    description: str,
    default: Any = ...,
    enum: list[Any] | tuple[Any, ...] | None = None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    items: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one property schema using the worker's supported schema keywords."""
    if not isinstance(kind, str) or kind not in {"object", "array", "string", "integer", "number", "boolean"}:
        raise ValueError(f"unsupported parameter type: {kind!r}")
    result: dict[str, Any] = {"type": kind, "description": description}
    if enum is not None:
        result["enum"] = list(enum)
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    if kind == "array":
        if items is None:
            raise ValueError("array parameters require an item schema")
        result["items"] = dict(items)
    elif items is not None:
        raise ValueError("items is valid only for array parameters")
    if default is not ...:
        result["default"] = default
    validate_parameter_schema(result)
    return result


def object_schema(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: tuple[str, ...] | list[str] = (),
    additional_properties: bool = False,
    description: str | None = None,
) -> dict[str, Any]:
    """Create the strict object schema consumed by component forms and checks."""
    result: dict[str, Any] = {
        "type": "object",
        "properties": {name: dict(schema) for name, schema in properties.items()},
        "required": list(required),
        "additionalProperties": additional_properties,
    }
    if description is not None:
        result["description"] = description
    validate_parameter_schema(result)
    return result


def validate_parameter_schema(schema: Mapping[str, Any]) -> None:
    """Validate the documented JSON Schema subset without third-party imports."""
    _validate_schema(schema, "params_schema", 0)


def write_static_manifest(
    package_directory: str | os.PathLike[str],
    entry_point_name: str,
    manifest: ComponentManifest,
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``__model_maven__/<entry-point-name>.json`` without importing code.

    Existing files are protected by default. Build tools should generate this
    file before wheel construction, not from the package's runtime import path.
    """
    if not isinstance(entry_point_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", entry_point_name
    ):
        raise ValueError("entry_point_name must match Python packaging entry-point naming rules")
    if not isinstance(manifest, ComponentManifest):
        raise TypeError("manifest must be a ComponentManifest")
    payload = manifest.to_dict()
    # Round-trip protects against custom mapping types and non-finite numbers.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    root = Path(package_directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("package_directory must be an existing regular directory")
    sidecar_directory = root / "__model_maven__"
    if sidecar_directory.is_symlink():
        raise ValueError("manifest directory must not be a symlink")
    sidecar_directory.mkdir(mode=0o755, exist_ok=True)
    if sidecar_directory.is_symlink() or not sidecar_directory.is_dir():
        raise ValueError("manifest directory is not a regular directory")
    target = sidecar_directory / f"{entry_point_name}.json"
    if target.is_symlink():
        raise ValueError("manifest target must not be a symlink")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return target


def _validate_component_id(value: str) -> str:
    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError("component_id must use namespace/name")
    namespace, name = value.split("/", 1)
    if not _SEGMENT.fullmatch(namespace) or not _SEGMENT.fullmatch(name):
        raise ValueError("component_id segments may contain ASCII letters, digits, '-' and '_'")
    return namespace


def _validate_entrypoint(value: str, label: str) -> None:
    if not isinstance(value, str) or value.count(":") != 1:
        raise ValueError(f"{label} must use module:callable")
    module, attribute = value.split(":", 1)
    parts = module.split(".") + attribute.split(".")
    if not module or not attribute or not all(_IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError(f"{label} must contain dotted Python identifiers")


def _validate_capabilities(value: Mapping[str, Any]) -> None:
    for name, capability in value.items():
        if not isinstance(name, str) or not name.strip() or any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ValueError("capability names must be non-empty strings")
        if isinstance(capability, str):
            state, reason, requires = capability, None, None
        elif isinstance(capability, Mapping):
            if set(capability) - {"state", "reason", "requires"}:
                raise ValueError(f"unsupported capability fields for {name}")
            state = capability.get("state")
            reason = capability.get("reason")
            requires = capability.get("requires")
        else:
            raise ValueError(f"capability {name} must be a state string or object")
        if not isinstance(state, str) or state not in _STATES:
            raise ValueError(f"unsupported capability state for {name}: {state!r}")
        if state != "supported" and (not isinstance(reason, str) or not reason):
            raise ValueError(f"capability {name} with state {state} requires a reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"capability {name} reason must be a string")
        if requires is not None and (
            not isinstance(requires, (list, tuple))
            or any(not isinstance(item, str) or not item for item in requires)
            or len(set(requires)) != len(requires)
        ):
            raise ValueError(f"capability {name} requires must be unique non-empty strings")


def _validate_dependencies(values: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("dependencies must be a list")
    result = []
    seen = set()
    for index, dependency in enumerate(values):
        if not isinstance(dependency, Mapping) or set(dependency) - {"component_id", "version", "optional"}:
            raise ValueError(f"dependencies[{index}] has unsupported fields")
        component_id = dependency.get("component_id")
        _validate_component_id(component_id)
        if component_id in seen:
            raise ValueError(f"dependencies contains duplicate component {component_id}")
        seen.add(component_id)
        version = dependency.get("version")
        if not _valid_semver_range(version):
            raise ValueError(f"dependencies[{index}].version is not a supported SemVer range")
        optional = dependency.get("optional", False)
        if not isinstance(optional, bool):
            raise ValueError(f"dependencies[{index}].optional must be boolean")
        result.append({"component_id": component_id, "version": version, "optional": optional})
    return result


def _valid_semver_range(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    for raw in value.split(","):
        match = re.fullmatch(r"(\^|~|>=|<=|>|<|=)?\s*(.+)", raw.strip())
        if not match:
            return False
        operator, version = match.groups()
        core_suffix = re.fullmatch(r"([^+-]+)(.*)", version.strip())
        if not core_suffix:
            return False
        core, suffix = core_suffix.groups()
        parts = core.split(".")
        if not 1 <= len(parts) <= 3 or any(not _SEMVER_PART.fullmatch(part) for part in parts):
            return False
        wildcard = any(part in {"x", "X", "*"} for part in parts)
        if suffix and not _SEMVER_SUFFIX.fullmatch(suffix):
            return False
        if wildcard and operator not in (None, "="):
            return False
        if wildcard and any(part in {"x", "X", "*"} for part in parts[:-1]):
            return False
    return True


def _validate_schema(schema: Mapping[str, Any], path: str, depth: int) -> None:
    allowed = {
        "type", "description", "default", "enum", "properties", "required",
        "additionalProperties", "items", "minimum", "maximum",
    }
    if depth > 16 or not isinstance(schema, Mapping) or set(schema) - allowed:
        raise ValueError(f"{path} has unsupported structure or exceeds 16 levels")
    kind = schema.get("type")
    if not isinstance(kind, str) or kind not in {"object", "array", "string", "integer", "number", "boolean"}:
        raise ValueError(f"{path}.type is missing or unsupported")
    if "description" in schema and not isinstance(schema["description"], str):
        raise ValueError(f"{path}.description must be a string")
    for bound in ("minimum", "maximum"):
        if bound in schema and (
            isinstance(schema[bound], bool)
            or not isinstance(schema[bound], (int, float))
            or not math.isfinite(schema[bound])
        ):
            raise ValueError(f"{path}.{bound} must be a finite number")
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        raise ValueError(f"{path}.minimum must not exceed maximum")
    if kind == "object":
        properties = schema.get("properties")
        if not isinstance(properties, Mapping) or len(properties) > 256:
            raise ValueError(f"{path}.properties must be an object with at most 256 entries")
        required = schema.get("required", [])
        if (
            not isinstance(required, (list, tuple))
            or any(not isinstance(name, str) for name in required)
            or len(set(required)) != len(required)
        ):
            raise ValueError(f"{path}.required must be a unique string list")
        if any(not isinstance(key, str) or not key for key in properties):
            raise ValueError(f"{path}.properties names must be non-empty strings")
        if any(not isinstance(key, str) or key not in properties for key in required):
            raise ValueError(f"{path}.required must refer to declared properties")
        if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
            raise ValueError(f"{path}.additionalProperties must be boolean")
        for name, child in properties.items():
            _validate_schema(child, f"{path}.properties.{name}", depth + 1)
    elif kind == "array":
        if not isinstance(schema.get("items"), Mapping):
            raise ValueError(f"{path}.items is required for arrays")
        _validate_schema(schema["items"], f"{path}.items", depth + 1)
    elif "items" in schema:
        raise ValueError(f"{path}.items is valid only for arrays")
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{path}.enum must be a non-empty list")
        value_schema = {key: value for key, value in schema.items() if key != "enum"}
        for index, value in enumerate(values):
            _validate_value(value, value_schema, f"{path}.enum[{index}]", depth)
    if "default" in schema:
        _validate_value(schema["default"], schema, f"{path}.default", depth)


def _validate_value(value: Any, schema: Mapping[str, Any], path: str, depth: int) -> None:
    kind = schema.get("type")
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }.get(kind, False)
    if not valid:
        raise ValueError(f"{path} does not match type {kind}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above maximum")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not in enum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if any(name not in value for name in required):
            raise ValueError(f"{path} is missing a required property")
        if schema.get("additionalProperties") is False and any(name not in properties for name in value):
            raise ValueError(f"{path} contains an undeclared property")
        for name, child_value in value.items():
            if name in properties:
                _validate_value(child_value, properties[name], f"{path}.{name}", depth + 1)
    elif isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            _validate_value(child, schema["items"], f"{path}[{index}]", depth + 1)
