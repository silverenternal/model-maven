#!/usr/bin/env python3
"""Measure exact request lookup against a synthetic 1024-run history."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-request-scale-") as directory:
    project = Path(directory)
    runs = project / ".model-maven/runs"
    runs.mkdir(parents=True)
    target = "tui-1234567890abcdefghijkl"
    target_id = None
    for index in range(1024):
        run_id = f"run-{index:05d}-fixture"
        path = runs / run_id
        path.mkdir()
        request_id = target if index == 511 else f"other-request-{index:08d}"
        if request_id == target:
            target_id = run_id
        (path / "run.json").write_text(json.dumps({"schema_version":1, "id":run_id,
            "operation":"train", "status":"completed", "started_at_ms":index,
            "request_id":request_id}))
    def run(*args):
        start = time.monotonic()
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                capture_output=True, timeout=120, check=True)
        return json.loads(result.stdout), len(result.stdout), round((time.monotonic() - start) * 1000, 2)
    report, lookup_bytes, lookup_ms = run("runs", "find-request", target)
    listing, list_bytes, list_ms = run("runs", "list")
    assert report["data"]["complete"] and report["data"]["matches"][0]["id"] == target_id, report
    assert len(report["data"]["matches"]) == 1 and len(listing["data"]["runs"]) == 1024
    assert lookup_bytes * 10 < list_bytes, (lookup_bytes, list_bytes)
    print(json.dumps({"runs":1024, "lookup_ms":lookup_ms, "lookup_bytes":lookup_bytes,
                      "list_ms":list_ms, "list_bytes":list_bytes, "scope":"synthetic warm local filesystem"}))
