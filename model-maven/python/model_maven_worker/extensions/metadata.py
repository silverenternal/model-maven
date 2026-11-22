"""Pure validation for versioned Model-Maven component manifests.

This module deliberately performs no entry-point discovery or import.  It is
kept dependency-free so static discovery can validate metadata before loading
third-party code.
"""

import json
import math
import re
import unicodedata
from pathlib import PurePosixPath


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMPONENT_SEGMENT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SEMVER_CORE_PART = re.compile(r"^(?:0|[1-9][0-9]*|[xX*])$")
_SEMVER_SUFFIX = re.compile(
    r"^(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_ENTRY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STATE_FORMAT_ID = re.compile(r"^[a-z][a-z0-9_.-]{1,127}$")
_SCHEMA_KEYS = {
    "type",
    "description",
    "default",
    "enum",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minimum",
    "maximum",
}
_MANIFEST_KEYS = {
    "schema_version",
    "namespace",
    "component_id",
    "component_version",
    "extension_api_version",
    "kind",
    "entrypoint",
    "description",
    "params_schema",
    "capabilities",
    "dependencies",
    "docs",
    "port_contracts",
    "renderer",
    "state_schema",
    "conformance_fixture",
    "package_source",
}
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
_CAPABILITY_STATES = {"supported", "unsupported", "unknown", "conditional"}
_MAX_DEPTH = 16


def _fail(path, message):
    raise ValueError(f"{path}: {message}")


def _component_id(value, path):
    if not isinstance(value, str) or value.count("/") != 1:
        _fail(path, "expected namespace/name")
    namespace, name = value.split("/", 1)
    if not _COMPONENT_SEGMENT.fullmatch(namespace) or not _COMPONENT_SEGMENT.fullmatch(name):
        _fail(path, "segments may contain only ASCII letters, digits, '-' and '_'")
    return namespace


def _entrypoint(value, path):
    if not isinstance(value, str) or value.count(":") != 1:
        _fail(path, "expected module:callable")
    module, attribute = value.split(":", 1)
    parts = module.split(".") + attribute.split(".")
    if not module or not attribute or not all(_IDENTIFIER.fullmatch(part) for part in parts):
        _fail(path, "only dotted Python identifiers are allowed")


def _json_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -(2**63) <= value <= 2**64 - 1
    )


def _control(char):
    return unicodedata.category(char) == "Cc"


def _semver_range(value):
    """Check the SemVer range forms used by the Rust semver::VersionReq."""
    if not isinstance(value, str) or not value.strip():
        return False
    for raw in value.split(","):
        comparator = raw.strip()
        if not comparator:
            return False
        match = re.fullmatch(r"(\^|~|>=|<=|>|<|=)?\s*(.+)", comparator)
        if not match:
            return False
        operator, version = match.groups()
        core_and_suffix = re.fullmatch(r"([^+-]+)(.*)", version.strip())
        if not core_and_suffix:
            return False
        core, suffix = core_and_suffix.groups()
        parts = core.split(".")
        if not 1 <= len(parts) <= 3 or any(not _SEMVER_CORE_PART.fullmatch(part) for part in parts):
            return False
        if suffix and not _SEMVER_SUFFIX.fullmatch(suffix):
            return False
        wildcard = any(part in {"x", "X", "*"} for part in parts)
        if wildcard and operator not in (None, "="):
            return False
        if wildcard and any(part in {"x", "X", "*"} for part in parts[:-1]):
            return False
    return True


def _validate_value(value, schema, path, depth):
    if depth > _MAX_DEPTH:
        _fail(path, "value nesting exceeds 16 levels")
    kind = schema.get("type")
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": _json_integer(value),
        "number": _json_number(value),
        "boolean": isinstance(value, bool),
    }.get(kind, False)
    if not valid:
        _fail(path, f"expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        _fail(path, "value is not in enum")
    if _json_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            _fail(path, f"below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            _fail(path, f"above maximum {schema['maximum']}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = next((key for key in value if key not in properties), None)
            if unknown is not None:
                _fail(f"{path}.{unknown}", "property is not declared")
        for name in schema.get("required", []):
            if name not in value:
                _fail(f"{path}.{name}", "required property is missing")
        for name, child in value.items():
            if name in properties:
                _validate_value(child, properties[name], f"{path}.{name}", depth + 1)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_value(child, schema["items"], f"{path}[{index}]", depth + 1)


def _validate_schema(schema, path, depth):
    if depth > _MAX_DEPTH:
        _fail(path, "schema nesting exceeds 16 levels")
    if not isinstance(schema, dict):
        _fail(path, "schema must be an object")
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown:
        _fail(path, f"unsupported keyword {sorted(unknown)[0]}")
    kind = schema.get("type")
    if not isinstance(kind, str) or kind not in {"object", "array", "string", "integer", "number", "boolean"}:
        _fail(path, "missing or unsupported type")
    if "description" in schema and not isinstance(schema["description"], str):
        _fail(path, "description must be a string")
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            _fail(path, "enum must be a non-empty array")
        for index, value in enumerate(values):
            _validate_value(value, {key: item for key, item in schema.items() if key != "enum"}, f"{path}.enum[{index}]", depth)
    for key in ("minimum", "maximum"):
        if key in schema and not _json_number(schema[key]):
            _fail(f"{path}.{key}", "must be a finite number")
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        _fail(path, "minimum must not exceed maximum")
    if kind == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            _fail(path, "object schema requires properties")
        if len(properties) > 256:
            _fail(path, "at most 256 properties are allowed")
        for name, child in properties.items():
            if not name or any(_control(char) for char in name):
                _fail(path, "property names must be non-empty and contain no controls")
            _validate_schema(child, f"{path}.properties.{name}", depth + 1)
        if "required" in schema:
            required = schema["required"]
            if not isinstance(required, list) or any(not isinstance(name, str) for name in required):
                _fail(path, "required must be a string array")
            if len(set(required)) != len(required) or any(name not in properties for name in required):
                _fail(path, "required names must be unique declared properties")
        if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
            _fail(path, "additionalProperties must be a boolean")
    elif kind == "array":
        if "items" not in schema:
            _fail(path, "array schema requires items")
        _validate_schema(schema["items"], f"{path}.items", depth + 1)
    if "default" in schema:
        _validate_value(schema["default"], schema, f"{path}.default", depth)


def validate_state_format(descriptor):
    """Validate an extension-owned state format descriptor at runtime.

    Extension state currently uses the worker's JSON-safe Schema subset. Tensor
    and arbitrary Python-object state continue to use ordinary unversioned
    callable hooks until a typed tensor-state contract is defined.
    """
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) - {"format_id", "version", "schema", "migrators"}
        or not {"format_id", "version", "schema"}.issubset(descriptor)
    ):
        _fail("state_format", "expected format_id, version, schema, and optional migrators")
    format_id = descriptor.get("format_id")
    if not isinstance(format_id, str) or not _STATE_FORMAT_ID.fullmatch(format_id):
        _fail("state_format.format_id", "expected a stable lowercase identifier")
    version = descriptor.get("version")
    if not isinstance(version, str) or not _SEMVER.fullmatch(version):
        _fail("state_format.version", "expected MAJOR.MINOR.PATCH without leading zeroes")
    schema = descriptor.get("schema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        _fail("state_format.schema", "top-level schema must be an object")
    _validate_schema(schema, "state_format.schema", 0)
    migrators = descriptor.get("migrators", {})
    if not isinstance(migrators, dict) or len(migrators) > 256:
        _fail("state_format.migrators", "must be an object with at most 256 routes")
    for route, entrypoint in migrators.items():
        if not isinstance(route, str) or route.count("->") != 1:
            _fail("state_format.migrators", "routes must use FROM_VERSION->TO_VERSION")
        from_version, to_version = route.split("->", 1)
        if not _SEMVER.fullmatch(from_version) or not _SEMVER.fullmatch(to_version):
            _fail(f"state_format.migrators.{route}", "versions must use MAJOR.MINOR.PATCH")
        if tuple(map(int, to_version.split("."))) <= tuple(map(int, from_version.split("."))):
            _fail(f"state_format.migrators.{route}", "migration routes must upgrade to a newer version")
        _entrypoint(entrypoint, f"state_format.migrators.{route}")
    try:
        json.dumps(descriptor, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        _fail("state_format", f"descriptor must be finite JSON data: {error}")
    return descriptor


def validate_state_value(state, descriptor):
    """Validate one JSON-safe state value against a validated format."""
    descriptor = validate_state_format(descriptor)
    try:
        _validate_value(state, descriptor["schema"], "state", 0)
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        _fail("state", f"value does not match the declared JSON state schema: {error}")
    return state


def _validate_capabilities(capabilities):
    if not isinstance(capabilities, dict):
        _fail("capabilities", "must be an object")
    for name, value in capabilities.items():
        if not name or any(_control(char) for char in name):
            _fail("capabilities", "names must be non-empty and contain no controls")
        if isinstance(value, str):
            state, reason, requires = value, None, None
        elif isinstance(value, dict):
            unknown = set(value) - {"state", "reason", "requires"}
            if unknown:
                _fail(f"capabilities.{name}", f"unsupported field {sorted(unknown)[0]}")
            state, reason, requires = value.get("state"), value.get("reason"), value.get("requires")
        else:
            _fail(f"capabilities.{name}", "must be a state string or object")
        if not isinstance(state, str) or state not in _CAPABILITY_STATES:
            _fail(f"capabilities.{name}.state", "unsupported state")
        if state != "supported" and (not isinstance(reason, str) or not reason):
            _fail(f"capabilities.{name}.reason", "non-supported states require a reason")
        if reason is not None and not isinstance(reason, str):
            _fail(f"capabilities.{name}.reason", "must be a string")
        if requires is not None:
            if not isinstance(requires, list) or any(not isinstance(item, str) or not item for item in requires):
                _fail(f"capabilities.{name}.requires", "must be a string array of non-empty names")
            if len(set(requires)) != len(requires):
                _fail(f"capabilities.{name}.requires", "entries must be unique")


def validate_manifest(manifest):
    """Validate a v1 component manifest; return the same object on success."""
    if not isinstance(manifest, dict):
        _fail("manifest", "top level must be an object")
    unknown = set(manifest) - _MANIFEST_KEYS
    if unknown:
        _fail("manifest", f"unsupported field {sorted(unknown)[0]}")
    schema_version = manifest.get("schema_version")
    api_version = manifest.get("extension_api_version")
    if (
        isinstance(schema_version, bool)
        or isinstance(api_version, bool)
        or schema_version != 1
        or api_version != 1
    ):
        _fail("manifest", "only schema and extension API version 1 are supported")
    component_id = manifest.get("component_id")
    namespace = _component_id(component_id, "component_id")
    if manifest.get("namespace") != namespace:
        _fail("namespace", "must match the component_id namespace")
    if not isinstance(manifest.get("component_version"), str) or not _SEMVER.fullmatch(manifest["component_version"]):
        _fail("component_version", "expected MAJOR.MINOR.PATCH without leading zeroes")
    if manifest.get("kind") not in _KINDS:
        _fail("kind", "unsupported component kind")
    _entrypoint(manifest.get("entrypoint"), "entrypoint")
    if "description" in manifest and (
        not isinstance(manifest["description"], str) or len(manifest["description"]) > 500
    ):
        _fail("description", "must be a string of at most 500 characters")
    schema = manifest.get("params_schema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        _fail("params_schema", "top-level schema must be an object")
    _validate_schema(schema, "params_schema", 0)
    _validate_capabilities(manifest.get("capabilities"))
    dependencies = manifest.get("dependencies", [])
    if not isinstance(dependencies, list):
        _fail("dependencies", "must be an array")
    seen = set()
    for index, dependency in enumerate(dependencies):
        path = f"dependencies[{index}]"
        if not isinstance(dependency, dict):
            _fail(path, "must be an object")
        if set(dependency) - {"component_id", "version", "optional"}:
            _fail(path, "supports only component_id, version, optional")
        dep_id = dependency.get("component_id")
        _component_id(dep_id, f"{path}.component_id")
        if dep_id in seen:
            _fail(path, f"duplicate dependency {dep_id}")
        seen.add(dep_id)
        requirement = dependency.get("version")
        if not _semver_range(requirement):
            _fail(f"{path}.version", "must be a valid SemVer range")
        if "optional" in dependency and not isinstance(dependency["optional"], bool):
            _fail(f"{path}.optional", "must be a boolean")
    for key in ("port_contracts", "renderer", "state_schema"):
        if key in manifest and not isinstance(manifest[key], dict):
            _fail(key, "must be an object")
    if "state_schema" in manifest:
        validate_state_format(manifest["state_schema"])
    if "docs" in manifest and not isinstance(manifest["docs"], (str, dict)):
        _fail("docs", "must be a string or object")
    if "conformance_fixture" in manifest:
        _entrypoint(manifest["conformance_fixture"], "conformance_fixture")
    if "package_source" in manifest:
        _validate_package_source(manifest["package_source"], manifest)
    return manifest


def _validate_package_source(source, manifest):
    if not isinstance(source, dict):
        _fail("package_source", "must be an object")
    allowed = {
        "distribution",
        "distribution_version",
        "entry_point",
        "manifest_file",
        "manifest_sha256",
        "distribution_root",
        "module_sources",
    }
    unknown = set(source) - allowed
    if unknown:
        _fail("package_source", f"unsupported field {sorted(unknown)[0]}")
    for field in ("distribution", "distribution_version", "entry_point"):
        value = source.get(field)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 256
            or any(_control(char) or char.isspace() for char in value)
        ):
            _fail(f"package_source.{field}", "must be a non-empty printable token")
    if not _ENTRY_NAME.fullmatch(source["entry_point"]):
        _fail("package_source.entry_point", "contains unsupported characters")
    manifest_file = source.get("manifest_file")
    path = PurePosixPath(manifest_file) if isinstance(manifest_file, str) else None
    if (
        path is None
        or not manifest_file
        or "\\" in manifest_file
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail("package_source.manifest_file", "must be a safe relative POSIX path")
    digest = source.get("manifest_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        _fail("package_source.manifest_sha256", "must be a lowercase SHA-256 digest")
    distribution_root = source.get("distribution_root")
    if not isinstance(distribution_root, str) or not PurePosixPath(distribution_root).is_absolute():
        _fail("package_source.distribution_root", "must be an absolute path")
    module_sources = source.get("module_sources")
    if not isinstance(module_sources, dict) or "entrypoint" not in module_sources:
        _fail("package_source.module_sources", "must contain entrypoint")
    if set(module_sources) - {"entrypoint", "fixture"}:
        _fail("package_source.module_sources", "only entrypoint and fixture are supported")
    if ("conformance_fixture" in manifest) != ("fixture" in module_sources):
        _fail("package_source.module_sources", "fixture fingerprint must match conformance_fixture")
    for role, fingerprint in module_sources.items():
        if not isinstance(fingerprint, dict) or set(fingerprint) != {"file", "sha256"}:
            _fail(f"package_source.module_sources.{role}", "must contain only file and sha256")
        relative = fingerprint["file"]
        source_path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            source_path is None
            or not relative
            or "\\" in relative
            or source_path.is_absolute()
            or not source_path.parts
            or any(part in {"", ".", ".."} for part in source_path.parts)
        ):
            _fail(f"package_source.module_sources.{role}.file", "must be a safe relative POSIX path")
        if not re.fullmatch(r"[0-9a-f]{64}", str(fingerprint["sha256"])):
            _fail(f"package_source.module_sources.{role}.sha256", "must be a lowercase SHA-256 digest")


def validate_parameters(value, schema, path="params"):
    """Validate parameter values using the same supported Schema subset."""
    _validate_value(value, schema, path, 0)
    return value
