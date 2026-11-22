#!/usr/bin/env python3
"""Run the CLI integration for hash-verified standalone restore and state access."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", help="accepted for the common QA runner; cargo builds the test binary")
    parser.add_argument("--python", default=sys.executable, help="interpreter used by the Rust CLI integration")
    args = parser.parse_args()
    environment = os.environ.copy()
    environment["MODEL_MAVEN_TEST_PYTHON"] = args.python
    command = [
        "cargo", "test", "--locked", "--test", "cli",
        "export_training_run_copies_verified_sources_and_restores_weights",
        "--", "--exact", "--nocapture",
    ]
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
