#!/usr/bin/env python3
"""Exercise persistent starter drafts across real TUI processes and a lock holder."""
import fcntl
import json
import os
import pty
import select
import signal
import time
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-draft-restart-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    def run(inputs):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input=inputs, capture_output=True, text=True, timeout=120, check=True)
    # Terminate after a field was accepted but before the form can finish.
    master, slave = pty.openpty()
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                             env=env, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    def wait_for(marker):
        data, deadline = b"", time.monotonic() + 30
        while marker.encode() not in data:
            assert time.monotonic() < deadline, (marker, data[-3000:])
            if select.select([master], [], [], .1)[0]: data += os.read(master, 65536)
    try:
        wait_for("Open workbench"); os.write(master, b"\n")
        wait_for("Model-Maven Workbench"); os.write(master, b"m\n")
        wait_for("Inputs and optional routes"); os.write(master, b"a=4;b=2;@a_encoder=b\n")
        wait_for("Output features"); os.write(master, b"3\n")
        wait_for("Custom model preview")
        os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
    finally:
        if child.poll() is None: os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
        os.close(master)
    draft = project / ".model-maven/scaffold-draft.json"
    assert json.loads(draft.read_text()) == {"specs": "a=4;b=2;@a_encoder=b", "output": "3"}
    original = draft.read_bytes()
    with (project / ".model-maven/scaffold-draft.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        competing = run("\nm\nx=9\nb\nq\n")
        assert "Draft recovery unavailable" in competing.stderr, competing.stderr
        assert draft.read_bytes() == original
    restored = run("\nm\n\n\ny\nq\n")
    assert "Current: a=4;b=2;@a_encoder=b" in restored.stderr, restored.stderr
    assert "Current: 3" in restored.stderr, restored.stderr
    assert '"a_encoder": "b"' in (project / "model_starter.py").read_text()
    assert not draft.exists()
    project = project / "corrupt-project"
    project.mkdir()
    (project / ".model-maven").mkdir()
    draft = project / ".model-maven/scaffold-draft.json"
    draft.write_bytes(b"{broken")
    invalid = run("\nm\nx=7\nb\nq\n")
    assert "Draft recovery unavailable" in invalid.stderr and draft.read_bytes() == b"{broken"
    print("Persistent draft survives a killed process, lock contention does not overwrite it, and successful creation clears it")
