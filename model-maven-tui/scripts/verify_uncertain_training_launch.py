#!/usr/bin/env python3
"""Lose a real detached launch receipt with a long synthetic run list."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-uncertain-training-") as directory:
    project = Path(directory) / "project"
    project.mkdir()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    wrapper = Path(directory) / "lost-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
mode = os.environ.get("LOST_TEST_MODE", "default")
if "--describe-interface" in args and mode in ("baseline", "no-spawn"):
    result = subprocess.run([{cli!r}] + args, capture_output=True)
    report = json.loads(result.stdout)
    report["data"]["capabilities"].pop("training.request_identity", None)
    sys.stdout.write(json.dumps(report)); sys.stderr.buffer.write(result.stderr); sys.exit(result.returncode)
if "runs" in args and "list" in args:
    marker = Path({str(Path(directory) / 'list-seen')!r})
    if mode == "baseline" and not marker.exists():
        marker.write_text("1"); print("invalid receipt"); sys.exit(0)
    result = subprocess.run([{cli!r}] + args, capture_output=True)
    report = json.loads(result.stdout)
    if report.get("status") == "success":
        report["data"]["runs"] = [{{"id": f"old-{{i}}", "status": "completed"}} for i in range(6)] + report["data"]["runs"]
    sys.stdout.write(json.dumps(report)); sys.stderr.buffer.write(result.stderr); sys.exit(result.returncode)
if "train" in args and "--detach" in args:
    if mode == "no-spawn": print("invalid receipt"); sys.exit(0)
    Path({str(Path(directory) / 'last-train-args.json')!r}).write_text(json.dumps(args))
    result = subprocess.run([{cli!r}] + args, capture_output=True)
    if result.returncode == 0:
        print("invalid receipt"); sys.exit(0)
    sys.stdout.buffer.write(result.stdout); sys.stderr.buffer.write(result.stderr); sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + args)
''')
    wrapper.chmod(0o755)
    def session(inputs, binary=cli, extra_env=None):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(binary)],
                              cwd=project, env=dict(env, **(extra_env or {})), input=inputs, text=True,
                              capture_output=True, timeout=180, check=True).stdout
    def command(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, text=True, capture_output=True, timeout=90, check=True)
        return json.loads(result.stdout)
    assert "Custom model project created" in session("\nm\nx=4\n2\ny\nq\n")
    output = session("\nr\ntrain\n\n\n\n1\n\nn\ny\nq\n", wrapper)
    runs = command("runs", "list")["data"]["runs"]
    actual = [item for item in runs if item["operation"] == "train"]
    assert len(actual) == 1, (output, runs)
    run_id = actual[0]["id"]
    assert "Checking saved runs; no automatic retry" in output, output
    assert "Matching start request" in output and run_id in output, output
    assert "old-0" not in output, output
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = command("runs", "attach", run_id, "--follow=false")["data"]["record"]["status"]
        if status not in ("running", "stopping"): break
        time.sleep(.25)
    assert status == "completed", status
    launch_args = json.loads((Path(directory) / "last-train-args.json").read_text())
    request_id = actual[0]["request_id"]
    assert "--request-id" in launch_args and launch_args[launch_args.index("--request-id") + 1] == request_id
    repeated = subprocess.run([cli, *launch_args], cwd=project, env=env, capture_output=True, text=True, timeout=120)
    repeated_report = json.loads(repeated.stdout)
    assert repeated.returncode != 0 and repeated_report["diagnostics"][0]["code"] == "RUN_REQUEST_ID_EXISTS", repeated_report
    assert repeated_report["data"]["run_id"] == run_id
    assert len([r for r in command("runs", "list")["data"]["runs"] if r["operation"] == "train"]) == 1
    blocked = session("\nr\ntrain\nq\n")
    assert "earlier training start has not been acknowledged" in blocked, blocked
    pending = session("\nj\nforget 1\ny\nb\nq\n")
    assert "Pending start 1" in pending and run_id in pending, pending
    fallback = session("\nr\ntrain\n\n\n\n1\n\nn\ny\nq\n", wrapper, {"LOST_TEST_MODE": "baseline"})
    runs = [item for item in command("runs", "list")["data"]["runs"] if item["operation"] == "train"]
    assert len(runs) == 2 and "Earlier run list was unavailable" in fallback, (fallback, runs)
    second = next(item["id"] for item in runs if item["id"] != run_id)
    assert second in fallback, fallback
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = command("runs", "attach", second, "--follow=false")["data"]["record"]["status"]
        if status not in ("running", "stopping"): break
        time.sleep(.25)
    assert status == "completed", status
    absent = session("\nr\ntrain\n\n\n\n1\n\nn\ny\nq\n", wrapper, {"LOST_TEST_MODE": "no-spawn"})
    assert "No new run visible yet" in absent and "does not prove" in absent, absent
    assert len([r for r in command("runs", "list")["data"]["runs"] if r["operation"] == "train"]) == 2
    ordered = command("runs", "list")["data"]["runs"]
    selected = next(index for index, item in enumerate(ordered, 1) if item["id"] == run_id) + 6
    resumed = session(f"\nj\n{selected}\nresume\n1\n\ny\nb\nq\n", wrapper)
    runs = [item for item in command("runs", "list")["data"]["runs"] if item["operation"] == "train"]
    children = [item for item in runs if item["parent_run_id"] == run_id]
    assert len(children) == 1 and "Matching start request" in resumed and children[0]["id"] in resumed, (resumed, runs)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = command("runs", "attach", children[0]["id"], "--follow=false")["data"]["record"]["status"]
        if status not in ("running", "stopping"): break
        time.sleep(.25)
    assert status == "completed", status
    pending = session("\nj\nforget 1\ny\nb\nq\n")
    assert "Pending start 1" in pending and children[0]["id"] in pending, pending
    print("Lost launch and resume receipts locate the matching real run; duplicates are rejected; legacy fallback stays conservative")
