#!/usr/bin/env python3
"""Exercise declared Sequential relationships in real preview diffs."""
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

for width in (40, 100):
    with tempfile.TemporaryDirectory(prefix="maven-sequential-diff-") as directory:
        project = Path(directory)
        (project / "fixture.py").write_text('''import torch
from torch import nn
from collections import OrderedDict
def model():
    return nn.Sequential(OrderedDict([
        ("first", nn.Linear(4, 48)),
        ("head", nn.Linear(32, 8)),
        ("tail", nn.ReLU()),
    ]))
def batch(): return torch.ones(2, 4)
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
            send(b"fixture:batch\n")
            until("FORWARD_FAILED")
            send(b"b7")
            until("in_features")
            send(b"48\n")
            until("Bound input content")
            pending.clear()
            send(b"d")
            frame = until("g diagram · n/p").decode(errors="replace")
            assert "Repair difference" in frame, frame
            assert "Sequential order · declaration" in frame, frame
            assert "= model.first → model.head" in frame, frame
            if width == 100:
                assert frame.count("= model.first → model.head") >= 2, frame
            if width == 40:
                pending.clear()
                send(b"\t")
                frame = until("g diagram · n/p").decode(errors="replace")
                assert "Sequential order · declaration" in frame, frame
                assert "= model.first → model.head" in frame, frame
            pending.clear()
            send(b"g")
            frame = until("g summary · n/p").decode(errors="replace")
            assert "Declared order diagram · partial" in frame, frame
            assert "= ╭─ model.first" in frame and "╰─▶ model.head [target]" in frame, frame
            if width == 100:
                assert frame.count("= ╭─ model.first") == 2, frame
            else:
                pending.clear()
                send(b"\t")
                frame = until("g summary · n/p").decode(errors="replace")
                assert "= ╭─ model.first" in frame and "╰─▶ model.head [target]" in frame, frame
            pending.clear()
            send(b"v")
            until("Repair evidence")
            send(b"v")
            until("Declared order diagram · partial")
            pending.clear()
            send(b"g")
            until("Repair difference · same target")
            until("Sequential order · declaration")
            pending.clear()
            send(b"b")
            until("Bound input content")
            send(b"b")
            until("Commit guarded variant?")
            send(b"\n")
            until("Model-Maven Workbench")
            assert not (project / ".model-maven/variants/working.json").exists()
            send(b"q")
            child.wait(timeout=10)
            assert child.returncode == 0
            print(f"{width}x24: declared real CLI relationships reviewed and cancellation preserved")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
            os.close(master)
