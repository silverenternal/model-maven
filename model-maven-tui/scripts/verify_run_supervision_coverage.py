#!/usr/bin/env python3
"""A completed observe run must not imply an unmeasured rule was checked."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-supervision-run-coverage-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    def session(inputs):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True).stdout
    def command(*args):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120, check=True)
        return json.loads(proc.stdout)["data"]
    output = session("\nm\nimage=3,4,4;tabular=5\n3\ny\n2\n\n\n"
                     "t\n2\n\nfused\nFused features\ny\nfinite\nobserve\n0\ny\n"
                     "r\ntrain\n\n\n\n1\n\ny\ny\nq\n")
    match = re.search(r"Run ID:\s*(\S+)", output)
    assert match, output
    run_id = match.group(1)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        detail = command("runs", "attach", run_id, "--follow=false")
        status = detail["record"]["status"]
        if status not in ("running", "stopping"):
            break
        time.sleep(0.25)
    assert status == "completed", detail
    saved = command("runs", "show", run_id)
    rule = saved["report"]["data"]["supervision"]["rules"][0]
    assert rule["mode"] == "observe" and rule["steps_observed"] == 0 and rule["element_count"] == 0, rule
    listing = command("runs", "list")["runs"]
    index = next(i for i, run in enumerate(listing, 1) if run["id"] == run_id)
    output = session(f"\nj\n{index}\nb\nb\nq\n")
    assert "measured steps: 0" in output and "No measurable data for this rule" in output, output
    print("Completed run visibly distinguishes an enabled rule from an actual measurement")
