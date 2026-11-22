#!/usr/bin/env python3
"""Run the real Linux/PyTorch producer-consumer artifact integration test."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", help="accepted for the common QA runner; cargo builds the test binary")
    parser.add_argument("--python", default=sys.executable, help="PyTorch interpreter used by the CLI integration")
    args = parser.parse_args()
    environment = os.environ.copy()
    environment["MODEL_MAVEN_TEST_PYTHON"] = args.python
    command = [
        "cargo",
        "test",
        "--locked",
        "--test",
        "run_artifacts",
        "finished_run_registers_checkpoint_with_hash_and_storage_category",
        "--",
        "--exact",
        "--nocapture",
    ]
    result = subprocess.run(command, env=environment, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
