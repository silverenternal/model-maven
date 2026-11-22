#!/usr/bin/env python3
"""Guard the Rust dependency direction between domain, ports, app, and outer layers."""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--root",
    type=Path,
    default=Path(__file__).resolve().parents[1],
    help="repository root to inspect (defaults to this checkout)",
)
ROOT = parser.parse_args().root.resolve()
RULES: dict[str, tuple[re.Pattern[str], ...]] = {
    "src/domain": (
        re.compile(r"\bcrate::(?:cli|application|app|adapters|presentation|ports)\b"),
    ),
    "src/ports": (
        re.compile(r"\bcrate::(?:application|app|adapters|presentation)\b"),
        re.compile(r"\b(?:eprintln|println)!\s*\("),
        re.compile(r"\b(?:std::io|IsTerminal)\b"),
    ),
    "src/application": (
        re.compile(r"\bcrate::(?:adapters|presentation)\b"),
        re.compile(r"\b(?:eprintln|println)!\s*\("),
        re.compile(r"\b(?:std::io|IsTerminal)\b"),
    ),
    "src/adapters": (
        re.compile(r"\bcrate::(?:application|app|presentation)\b"),
    ),
}

violations: list[str] = []
for directory, patterns in RULES.items():
    for path in sorted((ROOT / directory).rglob("*.rs")):
        if path.name == "replay_tests.rs":
            # This test intentionally composes concrete adapters to prove replay boundaries.
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("//"):
                continue
            for pattern in patterns:
                if pattern.search(line):
                    violations.append(
                        f"{path.relative_to(ROOT)}:{number}: forbidden dependency or terminal I/O: {line.strip()}"
                    )
                    break

if violations:
    print("Rust architecture boundary violations:", file=sys.stderr)
    print("\n".join(violations), file=sys.stderr)
    raise SystemExit(1)

print("Rust architecture boundaries passed.")
