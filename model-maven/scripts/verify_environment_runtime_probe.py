#!/usr/bin/env python3
"""Verify the opt-in CPU runtime smoke against the selected Python interpreter."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()

    command = [
        str(args.binary.resolve(strict=True)),
        "--python",
        args.python,
        "--json",
        "env",
        "probe",
        "--library",
        "torch",
        "--check-runtime",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240, check=False)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"expected a JSON response; exit={result.returncode}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        ) from error

    if result.returncode != 0 or report.get("status") != "success":
        raise RuntimeError(f"environment probe failed: {report!r}; stderr={result.stderr!r}")
    data = report.get("data") or {}
    probes = data.get("probes") or []
    if data.get("selected_by") != "explicit-python":
        raise RuntimeError(f"selected interpreter source was not explicit: {data.get('selected_by')!r}")
    if data.get("auto_installed") is not False or data.get("auto_bound") is not False:
        raise RuntimeError("runtime probe installed a package or bound a backend")
    if len(probes) != 1 or probes[0].get("status") != "imported":
        raise RuntimeError(f"PyTorch was not importable in the selected interpreter: {probes!r}")
    check = probes[0].get("runtime_check") or {}
    if (
        data.get("inspection") != "explicit-allowlisted-import-and-runtime-smoke"
        or check.get("status") != "passed"
        or check.get("profile") != "torch-cpu-tensor-arithmetic"
        or check.get("device") != "cpu"
        or check.get("result") != 5.0
        or probes[0].get("auto_bound") is not False
    ):
        raise RuntimeError(f"unexpected PyTorch runtime smoke result: {report!r}")
    print(json.dumps({
        "status": "verified",
        "python_executable": data.get("python_executable"),
        "torch_version": probes[0].get("distribution_version"),
        "profile": check.get("profile"),
        "device": check.get("device"),
        "result": check.get("result"),
        "auto_installed": data.get("auto_installed"),
        "auto_bound": data.get("auto_bound"),
        "uncovered": ["GPU/accelerator initialization", "user models/trainers", "OS isolation"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"environment runtime probe acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1)
