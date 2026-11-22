#!/usr/bin/env python3
"""Exercise preserved preview diagrams and real commit/recheck receipts."""
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

for width, case in ((40, "stable"), (100, "stable"), (40, "changing"), (40, "failing")):
    with tempfile.TemporaryDirectory(prefix="maven-diff-receipt-") as directory:
        project = Path(directory)
        (project / "fixture.py").write_text('''import torch
from torch import nn
from collections import OrderedDict
from pathlib import Path
class Tail(nn.Module):
    def forward(self, x):
        path = Path("failure_count.txt")
        if path.exists() and int(path.read_text()) >= 4:
            raise RuntimeError("intentional recheck failure")
        return x
def model():
    return nn.Sequential(OrderedDict([
        ("first", nn.Linear(4, 48)),
        ("head", nn.Linear(32, 8)),
        ("tail", Tail()),
    ]))
def step(name):
    path = Path(name)
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    return count
def stable(): return torch.ones(2, 4)
def changing(): return torch.full((2, 4), 1.0 if step("change_count.txt") <= 3 else 2.0)
def failing():
    step("failure_count.txt")
    return torch.ones(2, 4)
''', encoding="utf-8")
        master, slave = pty.openpty()
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))
        env = os.environ.copy()
        env.update(TERM="xterm-256color", NO_COLOR="1", MODEL_MAVEN_REDUCE_MOTION="1",
                   MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"))
        env.pop("COLUMNS", None)
        env.pop("TMUX", None)
        child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project,
                                 stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
        os.close(slave)
        pending = bytearray()

        def until(token, timeout=60):
            needle = token.encode()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                position = pending.find(needle)
                if position >= 0:
                    end = position + len(needle)
                    result = bytes(pending[:end])
                    del pending[:end]
                    return result
                if child.poll() is not None:
                    raise AssertionError(f"terminal exited {child.returncode}: {pending[-4000:]!r}")
                if select.select([master], [], [], 0.1)[0]:
                    pending.extend(os.read(master, 65536))
            raise AssertionError(f"{width} columns: missing {token!r}: {pending[-4000:]!r}")

        def send(data):
            os.write(master, data)

        try:
            until("Open workbench")
            send(b"\n1fixture:model\n")
            until("Completed")
            send(b"b2\n")
            until("Batch factory")
            send(f"fixture:{case}\n".encode())
            until("FORWARD_FAILED")
            send(b"b7")
            until("in_features")
            send(b"48\n")
            until("Bound input content")
            pending.clear()
            send(b"d")
            until("g diagram · n/p")
            send(b"g")
            until("Declared order diagram · partial")
            until("g summary · n/p")
            send(b"b")
            until("Bound input content")
            send(b"b")
            until("Commit guarded variant?")
            send(b"y")
            until("Guarded recheck")
            until({"stable": "Completed", "changing": "INPUT_CONTENT_MISMATCH", "failing": "FORWARD_FAILED"}[case])
            until("d Difference")
            pending.clear()
            send(b"d")
            until("Declared order diagram · partial")
            until("Historical preview")
            until("Historical · commit confirmed")
            expected = {"stable": "Recheck · CLI reports success", "changing": "Recheck · unverified or rejected", "failing": "Recheck · CLI reports forward failure"}[case]
            until(expected)
            until("model.head [target]")
            send(b"v")
            until("Repair evidence")
            until("Historical · commit confirmed")
            until(expected)
            send(b"b")
            until("Declared order diagram · partial")
            send(b"b")
            until("Guarded recheck")
            send(b"b")
            until("Model-Maven Workbench")
            assert (project / ".model-maven/variants/working.json").exists()
            send(b"0")
            until("Session receipt")
            until("d Difference")
            send(b"d")
            until("Declared order diagram · partial")
            until("Historical · commit confirmed")
            until(expected)
            until("model.head [target]")
            send(b"bb")
            until("Model-Maven Workbench")
            send(b"q")
            child.wait(timeout=10)
            assert child.returncode == 0
            print(f"{width}x24 {case}: real guarded result retained in the same diagram")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
            os.close(master)
