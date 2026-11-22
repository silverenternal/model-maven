#!/usr/bin/env python3
"""Keep the compatibility contract aligned with Rust and Python protocol constants."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/contracts/compatibility.json"
PYTHON_PROTOCOL = ROOT / "python/model_maven_worker/protocol.py"
PYTHON_WORKER = ROOT / "python/model_maven_worker/__init__.py"
RUST_CAPABILITY = ROOT / "src/domain/capability.rs"
RUST_RUNS = ROOT / "src/adapters/runs.rs"


def literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        if (
                            isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "frozenset"
                            and len(node.value.args) == 1
                        ):
                            try:
                                values[target.id] = frozenset(
                                    ast.literal_eval(node.value.args[0])
                                )
                            except (ValueError, TypeError):
                                continue
    return values


def fail(message: str) -> None:
    print(f"Compatibility contract violation: {message}", file=sys.stderr)
    raise SystemExit(1)


contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
python = literal_assignments(PYTHON_PROTOCOL)
rust_source = RUST_CAPABILITY.read_text(encoding="utf-8")
versions = {
    "worker_protocol": re.search(r"WORKER_PROTOCOL_VERSION: u32 = (\d+)", rust_source),
    "worker_api": re.search(r"WORKER_API_VERSION: u32 = (\d+)", rust_source),
}
run_schema_version = re.search(
    r"RUN_METADATA_SCHEMA_VERSION: u64 = (\d+)",
    RUST_RUNS.read_text(encoding="utf-8"),
)
if contract.get("schema_version") != 1:
    fail("compatibility.json schema_version must be 1")
for interface in contract.get("interfaces", []):
    identifier = interface.get("id")
    if identifier in versions:
        match = versions[identifier]
        if not match or int(match.group(1)) != interface.get("version"):
            fail(f"{identifier} version differs from Rust capability constants")

if python.get("PROTOCOL_VERSION") != contract["interfaces"][0]["version"]:
    fail("Python worker protocol differs from compatibility.json")
if python.get("WORKER_API_VERSION") != contract["interfaces"][1]["version"]:
    fail("Python worker API differs from compatibility.json")
if python.get("STATE_FORMAT_VERSIONS") != {
    item["id"]: item["version"] for item in contract.get("state_formats", [])
}:
    fail("Python state format versions differ from compatibility.json")
host_state_formats = contract.get("host_state_formats", [])
if len(host_state_formats) != 1 or host_state_formats[0].get("id") != "run_record":
    fail("compatibility.json must declare the Rust run_record host state format")
if not run_schema_version or int(run_schema_version.group(1)) != host_state_formats[0].get("version"):
    fail("Rust run metadata version differs from compatibility.json")
operations = python.get("OPERATION_FEATURES")
if not isinstance(operations, dict):
    fail("Python worker operation registry is missing")
expected_operations = {
    identifier: list(features) for identifier, features in operations.items()
}
if contract.get("operations") != expected_operations:
    fail("Python operation registry differs from compatibility.json")
worker_tree = ast.parse(PYTHON_WORKER.read_text(encoding="utf-8"), filename=str(PYTHON_WORKER))
dispatch = next(
    (node for node in worker_tree.body if isinstance(node, ast.FunctionDef) and node.name == "dispatch"),
    None,
)
if dispatch is None:
    fail("Python worker dispatch function is missing")
dispatched_operations = set()
for node in ast.walk(dispatch):
    if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
        continue
    if node.left.id != "operation":
        continue
    for comparator in node.comparators:
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
            dispatched_operations.add(comparator.value)
        elif isinstance(comparator, (ast.Set, ast.List, ast.Tuple)):
            dispatched_operations.update(
                item.value
                for item in comparator.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
if dispatched_operations != set(operations):
    missing = sorted(set(operations) - dispatched_operations)
    unregistered = sorted(dispatched_operations - set(operations))
    fail(f"operation registry/dispatcher differ; missing routes={missing}, unregistered routes={unregistered}")
host_features = python.get("HOST_FEATURES")
if not isinstance(host_features, frozenset):
    fail("Python host capability registry is missing")
registered_features = frozenset(
    feature for features in operations.values() for feature in features
) | host_features
if registered_features != frozenset(contract.get("required_features", [])):
    fail("registered operation/host features differ from compatibility.json")
if len(contract.get("required_features", [])) != len(set(contract.get("required_features", []))):
    fail("compatibility.json contains duplicate required features")

print("Compatibility contract passed: Rust/Python protocol, API, state formats and required features match.")
