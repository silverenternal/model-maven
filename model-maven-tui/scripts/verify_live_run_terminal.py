#!/usr/bin/env python3
"""Exercise a real running training job in the full-screen PTY live view."""
import fcntl
import json
import os
from pathlib import Path
import pty
import re
import select
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
    (project / "fixture.py").write_text('''import time, torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)
    def forward(self, x): return self.head(x)
def model(): return Model()
def batch():
    return {"args": [], "kwargs": {"x": torch.ones(2, 4)}, "loss_inputs": {"target": torch.zeros(2, 2)}}
def step(model, batch, context):
    time.sleep(.5)
    return {"loss": (model(**batch["kwargs"]) - batch["loss_inputs"]["target"]).square().mean()}
batch.MODEL_MAVEN_STATELESS = True
step.MODEL_MAVEN_STATELESS = True
''')
    env = dict(os.environ, TERM="xterm-256color", NO_COLOR="1", MODEL_MAVEN_REDUCE_MOTION="1",
               MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    env.pop("COLUMNS", None)
    env.pop("TMUX", None)
    def command(*args, check=True):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        if check: assert result.returncode == 0, report
        return report
    command("init", "--model", "fixture:model", "--batch", "fixture:batch")
    started = command("train", "--step", "fixture:step", "--steps", "120", "--checkpoint-every", "10", "--detach")
    run_id = started["data"]["run_id"]
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 120, 0, 0))
    child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project, env=env,
                             stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    pending, transcript = bytearray(), bytearray()
    def wait(token, timeout=30):
        needle, deadline = token.encode(), time.monotonic() + timeout
        while time.monotonic() < deadline:
            position = pending.find(needle)
            if position >= 0:
                del pending[:position + len(needle)]
                return
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-4000:]!r}")
            if select.select([master], [], [], .1)[0]:
                chunk = os.read(master, 65536)
                pending.extend(chunk); transcript.extend(chunk)
        raise AssertionError(f"missing {token!r}: {pending[-4000:]!r}")
    def send(value): os.write(master, value)
    try:
        wait("Open workbench"); send(b"\nj")
        wait("Training runs"); send(b"b")
        wait("Run number, refresh"); send((run_id + "\n").encode())
        wait("Run status"); wait(run_id); send(b"b")
        wait("refresh, watch, stop"); send(b"\x15watch\n")
        wait("Live run status"); wait("Latest progress", timeout=30)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            steps = {int(value) for value in re.findall(rb'"step":\s*(\d+)', bytes(transcript))}
            if len(steps) >= 2: break
            if select.select([master], [], [], .2)[0]:
                chunk = os.read(master, 65536); pending.extend(chunk); transcript.extend(chunk)
        assert len(steps) >= 2, transcript[-5000:]
        pending.clear()
        send(b"r")
        wait("Latest progress", timeout=10)
        send(b"b")
        wait("Run status")
        send(b"b"); wait("refresh, watch, stop"); send(b"\x15b\n")
        wait("Training runs"); send(b"b")
        wait("Run number, refresh"); send(b"\x15b\n")
        wait("Model-Maven Workbench"); send(b"q")
        child.wait(timeout=15)
        assert child.returncode == 0
        record = command("runs", "show", run_id)["data"]["record"]
        assert record["status"] in ("running", "stopping"), record
        print(f"PTY live view observed steps {min(steps)}..{max(steps)}; leaving kept run {run_id} active")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
        command("runs", "stop", run_id, check=False)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            record = command("runs", "show", run_id, check=False)["data"].get("record", {})
            if record.get("status") not in ("running", "stopping"): break
            time.sleep(.2)
        if record.get("status") in ("running", "stopping"):
            command("runs", "stop", run_id, "--force", check=False)
            raise AssertionError("fixture training did not stop within 30 seconds")
