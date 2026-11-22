#!/usr/bin/env python3
"""Select a real graph module and apply a guarded source edit in a PTY."""
import fcntl
import os
import pty
import select
import struct
import subprocess
import tempfile
import termios
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-source-graph-") as directory:
    project = Path(directory)
    source = project / "fixture.py"
    source.write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)
    def forward(self, x): return self.head(x)
def model(): return Model()
def batch(): return torch.ones(2, 4)
''')
    subprocess.run([cli, "--json", "--no-interactive", "init", "--model", "fixture:model", "--batch", "fixture:batch"], cwd=project, check=True, capture_output=True)
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    env = os.environ.copy()
    env.update(TERM="xterm-256color", NO_COLOR="1", MODEL_MAVEN_REDUCE_MOTION="1", MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    env.pop("TMUX", None)
    child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    pending = bytearray()
    def until(token, timeout=60):
        needle, deadline = token.encode(), time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos = pending.find(needle)
            if pos >= 0:
                end = pos + len(needle)
                result = bytes(pending[:end])
                del pending[:end]
                return result.decode(errors="replace")
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-4000:]!r}")
            if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-4000:]!r}")
    def send(value): os.write(master, value)
    try:
        until("Open workbench")
        send(b"\n1fixture:model\n")
        until("Completed")
        send(b"b6")
        until("S:src")
        send(b"\x1b[Bs")
        until("Edit project Python source")
        until("Graph selection")
        send(b"b")
        until("add, set, replace")
        send(b"set\n")
        until("Source layer targets")
        send(b"b")
        until("Graph selection in current source model")
        details = until("inspected separately")
        assert "fixture:model" in details and "model.head" in details and "Linear" in details, details
        send(b"b")
        until("Continue? Type y")
        send(b"y\n")
        until("Editable layer parameters")
        send(b"b")
        until("Layer parameters")
        send(b'{"bias":false}\n')
        until("Source preview")
        until("fixture.py (candidate)")
        send(b"b")
        until("Type apply to modify source files")
        send(b"apply\n")
        until("Post-save verification")
        until('"status":"passed"')
        send(b"b")
        until("Model-Maven Workbench")
        assert "bias=False" in source.read_text()
        send(b"q")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("Graph-selected module is reidentified in the source model, confirmed, edited and revalidated")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
