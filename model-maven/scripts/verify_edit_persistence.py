#!/usr/bin/env python3
"""Run focused real CLI and checkpoint tests for edit persistence acceptance."""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMANDS = [
    [
        "cargo",
        "test",
        "--locked",
        "--test",
        "cli",
        "layer_variant_crud_is_persistent_undoable_and_transactional",
        "--",
        "--exact",
        "--nocapture",
    ],
    [
        "cargo",
        "test",
        "--locked",
        "--test",
        "cli",
        "workbench_layer_crud_keeps_model_and_batch_context_and_reloads_variant",
        "--",
        "--exact",
        "--nocapture",
    ],
    ["cargo", "test", "--locked", "--test", "edit_providers", "--", "--nocapture"],
    ["cargo", "test", "--locked", "--test", "edit_migration", "--", "--nocapture"],
]


def main() -> int:
    for command in COMMANDS:
        print("$ " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            print(f"FAILED (exit {completed.returncode})", file=sys.stderr, flush=True)
            return completed.returncode
    print("Edit persistence and migration acceptance passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
