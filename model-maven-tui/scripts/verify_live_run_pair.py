#!/usr/bin/env python3
"""Real detached training plus a full-screen PTY live-status session (Unix)."""
import json
import os
import fcntl
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import tempfile
import termios
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-live-run-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"), TERM="xterm-256color", NO_COLOR="1")
    def command(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=60)
        report = json.loads(result.stdout)
        assert result.returncode == 0, report
        return report
    args = ("scaffold", "--input", "x=4", "--output-dim", "2")
    plan = command(*args)
    command(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    (project / "slow_step.py").write_text("import time\nimport model_starter\ndef step(model, batch, context):\n    time.sleep(1)\n    return model_starter.step(model, batch, context)\nstep.MODEL_MAVEN_STATELESS = True\n")
    run = command("train", "--step", "slow_step:step", "--steps", "30", "--detach")
    run_id = run["data"]["run_id"]
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project, env=env,
                             stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    captured = bytearray()
    def wait_for(marker, timeout=20):
        deadline = time.monotonic() + timeout
        start = len(captured)
        needle = marker.encode()
        while needle not in captured[start:]:
            assert time.monotonic() < deadline, (marker, bytes(captured[-12000:]).decode(errors="replace"))
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                captured.extend(os.read(master, 65536))
        return bytes(captured[start:]).decode(errors="replace")
    try:
        wait_for("Enter")
        os.write(master, b"\r")
        wait_for("Model-Maven Workbench")
        os.write(master, b"j")
        wait_for(run_id)
        os.write(master, b"b")
        wait_for("Run number")
        os.write(master, b"1\r")
        wait_for("Run status")
        os.write(master, b"b")
        wait_for("refresh, watch")
        os.write(master, b"\x15watch\r")
        wait_for("Live run status")
        wait_for("Latest progress")
        wait_for("Latest progress", timeout=8)
        os.write(master, b"b")
        wait_for("Run status")
        current = command("runs", "attach", run_id, "--follow=false")
        assert current["data"]["record"]["status"] == "running", current
        print("Live status refreshed repeatedly; leaving watch kept detached training running")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        os.close(master)
        subprocess.run([cli, "--json", "--no-interactive", "runs", "stop", run_id], cwd=project,
                       env=env, capture_output=True, timeout=60)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if command("runs", "attach", run_id, "--follow=false")["data"]["record"]["status"] not in ("running", "stopping"):
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("training did not stop before temporary project cleanup")
