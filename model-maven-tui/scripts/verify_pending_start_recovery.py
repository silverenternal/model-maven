#!/usr/bin/env python3
"""Kill the real TUI while a launch receipt is in flight, then reopen it."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
with tempfile.TemporaryDirectory(prefix="maven-pending-recovery-") as directory:
    base = Path(directory)
    project = base / "project"
    project.mkdir()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    wrapper = base / "delay-cli.py"
    ready = base / "ready.json"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
if os.environ.get("MALFORMED_LOOKUP") == "1" and "runs" in args and "list" in args:
    print("invalid receipt"); sys.exit(0)
if os.environ.get("MALFORMED_LOOKUP") == "1" and "find-request" in args:
    result = subprocess.run([{cli!r}] + args, capture_output=True)
    report = json.loads(result.stdout)
    report["data"]["unreadable"] = 1
    report["data"]["complete"] = True
    print(json.dumps(report)); sys.exit(0)
if "train" in args and "--detach" in args:
    if os.environ.get("HANG_BEFORE_LAUNCH") == "1":
        Path({str(ready)!r}).write_text(json.dumps({{"pid":os.getpid(),"before_launch":True}}))
        time.sleep(90); sys.exit(0)
    result = subprocess.run([{cli!r}] + args, capture_output=True)
    if result.returncode != 0:
        sys.stdout.buffer.write(result.stdout); sys.stderr.buffer.write(result.stderr); sys.exit(result.returncode)
    Path({str(ready)!r}).write_text(json.dumps({{"pid":os.getpid(),"receipt":json.loads(result.stdout)}}))
    time.sleep(90)
    sys.stdout.buffer.write(result.stdout); sys.exit(0)
os.execv({cli!r}, [{cli!r}] + args)
''')
    wrapper.chmod(0o755)
    def session(inputs, binary=cli, extra_env=None):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(binary)], cwd=project,
                              env=dict(env, **(extra_env or {})), input=inputs, text=True, capture_output=True,
                              timeout=180, check=True).stdout
    def command(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, text=True, capture_output=True, timeout=90, check=True)
        return json.loads(result.stdout)
    assert "Custom model project created" in session("\nm\nx=4\n2\ny\nq\n")
    process = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)],
                               cwd=project, env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.stdin.write(b"\nr\ntrain\n\n\n\n1\n\nn\ny\n")
    process.stdin.flush()
    wrapper_pid = None
    try:
        deadline = time.monotonic() + 90
        while not ready.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(process.communicate())
            time.sleep(.1)
        assert ready.exists(), "training launch did not reach delayed receipt"
        info = json.loads(ready.read_text())
        wrapper_pid = info["pid"]
        run_id = info["receipt"]["data"]["run_id"]
        request_id = info["receipt"]["data"]["request_id"]
        marker = project / ".model-maven/tui-pending-starts" / request_id
        assert marker.is_file()
        process.kill()
        process.wait(timeout=10)
        os.kill(wrapper_pid, signal.SIGTERM)
        wrapper_pid = None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            record = command("runs", "show", run_id)["data"]["record"]
            if record["status"] not in ("running", "stopping"): break
            time.sleep(.2)
        assert record["status"] == "completed", record
        blocked = session("\nr\ntrain\nq\n")
        assert "earlier training start has not been acknowledged" in blocked, blocked
        malformed = session("\nj\nb\nq\n", wrapper, {"MALFORMED_LOOKUP":"1"})
        assert "Request lookup is unavailable or omitted" in malformed and marker.exists(), malformed
        damaged = project / ".model-maven/runs/run-123-broken"
        damaged.mkdir()
        (damaged / "run.json").write_text("invalid run metadata")
        reopened = session(f"\nj\n{run_id}\nb\nb\nq\n")
        assert "Full run list is unavailable" in reopened and "Run status" in reopened and run_id in reopened, reopened
        checkpoint = project / ".model-maven/runs" / run_id / "checkpoints/latest.json"
        saved_checkpoint = checkpoint.read_bytes()
        checkpoint.write_text("invalid checkpoint metadata")
        output = session("\nj\nforget 1\ny\nb\nq\n")
        assert "Pending start 1" in output and "Matching run" in output and run_id in output, output
        assert "Full run list is unavailable" in output and "other matches may exist" in output, output
        assert not marker.exists()
        checkpoint.write_bytes(saved_checkpoint)
        (damaged / "run.json").unlink()
        damaged.rmdir()
        assert command("runs", "show", run_id)["data"]["record"]["status"] == "completed"
        ready.unlink()
        process = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)],
                                   cwd=project, env=dict(env, HANG_BEFORE_LAUNCH="1"), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.stdin.write(b"\nr\ntrain\n\n\n\n1\n\nn\ny\n")
        process.stdin.flush()
        deadline = time.monotonic() + 90
        while not ready.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(process.communicate())
            time.sleep(.1)
        assert ready.exists(), "pre-launch interruption did not reach wrapper"
        info = json.loads(ready.read_text())
        assert info["before_launch"] is True
        wrapper_pid = info["pid"]
        process.kill()
        process.wait(timeout=10)
        os.kill(wrapper_pid, signal.SIGTERM)
        wrapper_pid = None
        output = session("\nj\nforget 1\ny\nb\nq\n")
        assert "Pending start 1" in output and "outcome remains unknown" in output, output
        assert len(command("runs", "list")["data"]["runs"]) == 1
        print("Killed TUI during lost receipt: reopening identifies the real run and requires explicit acknowledgement")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if wrapper_pid is not None:
            try: os.kill(wrapper_pid, signal.SIGTERM)
            except ProcessLookupError: pass
