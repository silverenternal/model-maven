#!/usr/bin/env python3
"""Run fixture suites for registered components and the packaged Provider flow."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    environment = os.environ.copy()
    environment["MODEL_MAVEN_TEST_PYTHON"] = args.python
    conformance = subprocess.run(
        ["cargo", "test", "--locked", "--test", "extensions_conformance", "--", "--nocapture"],
        env=environment,
        check=False,
    )
    if conformance.returncode:
        return conformance.returncode
    registered = subprocess.run(
        ["cargo", "test", "--locked", "--test", "component_transform_conformance", "--", "--nocapture"],
        env=environment,
        check=False,
    )
    if registered.returncode:
        return registered.returncode
    lifecycle = subprocess.run(
        [
            "cargo",
            "test",
            "--locked",
            "--test",
            "edit_providers",
            "--test",
            "extensions_state",
            "--",
            "--nocapture",
        ],
        env=environment,
        check=False,
    )
    if lifecycle.returncode:
        return lifecycle.returncode
    component_state = subprocess.run(
        ["cargo", "test", "--locked", "--test", "extensions_component_state", "--", "--nocapture"],
        env=environment,
        check=False,
    )
    if component_state.returncode:
        return component_state.returncode
    trainer = subprocess.run(
        ["cargo", "test", "--locked", "--test", "extensions_trainer_conformance", "--", "--nocapture"],
        env=environment,
        check=False,
    )
    if trainer.returncode:
        return trainer.returncode
    worker_timeout = subprocess.run(
        [
            "cargo",
            "test",
            "--locked",
            "--test",
            "cli",
            "timeout_",
            "--",
            "--nocapture",
        ],
        env=environment,
        check=False,
    )
    if worker_timeout.returncode:
        return worker_timeout.returncode
    training_contracts = subprocess.run(
        [
            args.python,
            "scripts/verify_external_loop.py",
            "--binary",
            str(args.binary),
            "--python",
            args.python,
        ],
        check=False,
    )
    if training_contracts.returncode:
        return training_contracts.returncode
    provider = subprocess.run(
        [
            args.python,
            "scripts/verify_extension_conformance.py",
            "--binary",
            str(args.binary),
            "--python",
            args.python,
        ],
        check=False,
    )
    return provider.returncode


if __name__ == "__main__":
    raise SystemExit(main())
