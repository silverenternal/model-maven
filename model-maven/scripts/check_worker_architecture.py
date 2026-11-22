#!/usr/bin/env python3
"""Check worker module direction and Rust embedded asset coverage."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python/model_maven_worker"
ASSET_SOURCE = ROOT / "src/adapters/worker_assets.rs"

ALLOWED_DOMAINS = {
    "protocol": set(),
    "runtime": {"protocol", "runtime"},
    "environment": {"protocol", "environment"},
    "data": {"protocol", "data", "tensor_inputs"},
    "model": {"protocol", "model", "tensor_inputs"},
    # Bounded identities and codecs for already-bound model forward inputs.
    # This shared value layer stays below both model probing and data tooling.
    "tensor_inputs": {"protocol", "tensor_inputs"},
    "edit": {"protocol", "edit", "model"},
    # Training consumes versioned extension state contracts while restoring
    # user callables; extension metadata remains pure and has no train import.
    "train": {"protocol", "train", "environment", "model", "data", "run", "extensions"},
    "run": {"protocol", "run"},
    "extensions": {"protocol", "extensions", "environment", "model", "edit", "data"},
}
ASSET_PATTERN = re.compile(
    r'path:\s*"([^"]+)"\s*,\s*contents:\s*include_str!\("../../([^"]+)"\)'
)

violations: list[str] = []
asset_entries = ASSET_PATTERN.findall(ASSET_SOURCE.read_text(encoding="utf-8"))
asset_destinations = [destination for destination, _ in asset_entries]
if len(asset_destinations) != len(set(asset_destinations)):
    violations.append("worker_assets.rs contains duplicate staged destinations")
asset_sources = [source for _, source in asset_entries]
package_sources = {
    path.relative_to(ROOT).as_posix()
    for path in PACKAGE.rglob("*.py")
    if path.is_file()
}
embedded_package_sources = {source for source in asset_sources if source.startswith("python/model_maven_worker/")}
for missing in sorted(package_sources - embedded_package_sources):
    violations.append(f"Python worker module is missing from embedded assets: {missing}")
for stale in sorted(embedded_package_sources - package_sources):
    violations.append(f"Embedded worker asset has no Python source file: {stale}")
for destination, source in asset_entries:
    if not (ROOT / source).is_file():
        violations.append(f"Embedded asset source does not exist: {destination} -> {source}")
if "python/worker.py" not in asset_sources:
    violations.append("worker.py compatibility launcher is missing from embedded assets")

modules: dict[str, Path] = {}
for path in sorted(PACKAGE.rglob("*.py")):
    relative = path.relative_to(PACKAGE)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    module = ".".join(["model_maven_worker", *parts])
    modules[module] = path

graph: dict[str, set[str]] = defaultdict(set)
for module, path in modules.items():
    relative = path.relative_to(PACKAGE)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        continue  # The package initializer is the composition root.
    domain = parts[0]
    allowed = ALLOWED_DOMAINS.get(domain)
    if allowed is None:
        violations.append(f"Unknown worker domain in {path.relative_to(ROOT)}")
        continue
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "model_maven_worker" or alias.name.startswith("model_maven_worker."):
                    target = alias.name
                    if target == "model_maven_worker":
                        violations.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: {domain} imports the worker composition root"
                        )
                        continue
                    target_parts = target.split(".")
                    target_domain = target_parts[1] if len(target_parts) > 1 else ""
                    if target_domain not in allowed:
                        violations.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: {domain} imports forbidden worker domain {target}"
                        )
                    if target in modules:
                        graph[module].add(target)
                    elif not any(name.startswith(target + ".") for name in modules):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: unresolved worker import {target}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                current_package = module.split(".")[:-1]
                if path.name == "__init__.py":
                    current_package = module.split(".")
                keep = max(0, len(current_package) - (node.level - 1))
                base = current_package[:keep]
                suffix = node.module.split(".") if node.module else []
                target = ".".join([*base, *suffix])
            elif node.module and node.module.startswith("model_maven_worker"):
                target = node.module
        if not target or not target.startswith("model_maven_worker."):
            continue
        target_parts = target.split(".")
        target_domain = target_parts[1] if len(target_parts) > 1 else ""
        if target_domain not in allowed:
            violations.append(
                f"{path.relative_to(ROOT)}:{node.lineno}: {domain} imports forbidden worker domain {target}"
            )
        if target in modules:
            graph[module].add(target)
        elif not any(name.startswith(target + ".") for name in modules):
            violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: unresolved worker import {target}")

visiting: set[str] = set()
visited: set[str] = set()

def visit(module: str, chain: list[str]) -> None:
    if module in visiting:
        cycle = " -> ".join([*chain, module])
        violations.append(f"Worker module import cycle: {cycle}")
        return
    if module in visited:
        return
    visiting.add(module)
    for dependency in sorted(graph.get(module, ())):
        visit(dependency, [*chain, module])
    visiting.remove(module)
    visited.add(module)

for module in sorted(modules):
    visit(module, [])

if violations:
    print("Worker architecture violations:", file=sys.stderr)
    print("\n".join(sorted(set(violations))), file=sys.stderr)
    raise SystemExit(1)

print(f"Worker architecture passed: {len(modules)} modules, embedded asset coverage complete, imports follow domain direction.")
