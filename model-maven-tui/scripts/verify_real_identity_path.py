#!/usr/bin/env python3
"""Exercise partial tensor identity evidence in real full-screen terminal sessions."""
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

for width, graphics, mode in [(40, "cells", "stable"), (100, "pixels", "stable"), (40, "cells", "cloned"), (100, "pixels", "mutating"), (40, "cells", "fanout"), (100, "pixels", "fanout"), (40, "cells", "fanout_mono"), (40, "cells", "shared")]:
    with tempfile.TemporaryDirectory(prefix="maven-identity-terminal-") as directory:
        project = Path(directory)
        (project / "fixture.py").write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.src = nn.Linear(4, 48)
        self.identity = nn.Identity()
        self.head = nn.Linear(32, 8)
        if mode == "shared": self.again = self.src
        if mode == "fanout": self.branches = nn.ModuleList([nn.Linear(48, 48) for _ in range(32)])
        self.mode = mode
    def forward(self, x):
        value = self.identity(self.src(x))
        if self.mode == "mutating": value.add_(1)
        if self.mode == "cloned": value = value.clone()
        if self.mode == "fanout":
            for branch in self.branches: branch(value)
        return self.head(value)
def stable(): return Model("stable")
def mutating(): return Model("mutating")
def cloned(): return Model("cloned")
def fanout(): return Model("fanout")
def fanout_mono(): return Model("fanout")
def shared(): return Model("shared")
def batch(): return torch.ones(2, 4)
''', encoding="utf-8")
        master, slave = pty.openpty()
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))
        env = os.environ.copy()
        env.update(TERM="xterm-256color", MODEL_MAVEN_GRAPHICS=graphics, MODEL_MAVEN_REDUCE_MOTION="1",
                   MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"))
        env.pop("NO_COLOR", None)
        if mode == "fanout_mono": env["NO_COLOR"] = "1"
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
            send(f"\n1fixture:{mode}\n".encode())
            until("Completed")
            send(b"b2\n")
            until("Batch factory")
            send(b"fixture:batch\n")
            until("FORWARD_FAILED")
            send(b"b8")
            until("model.head")
            send(b"F")
            until("2 object matches · partial" if mode in ("stable", "fanout", "fanout_mono", "shared") else "Source not recorded")
            if mode == "stable":
                until("1/2 ←→ model.src")
                pending.clear()
                send(b"\x1b[C")
                until("2/2 ←→ model.identity")
                pending.clear()
                send(b"FF")
                until("2/2 ←→ model.identity")
                pending.clear()
                send(b"\n")
                until("Candidate call · partial match")
                until("Observed call · model.identity")
                until("1/1 ←→ model.src")
                pending.clear()
                send(b"F")
                until("model.identity   ·   Identity")
                pending.clear()
                send(b"F")
                until("Candidate call · partial match")
                until("Observed call · model.identity")
                until("1/1 ←→ model.src")
                pending.clear()
                send(b"\n")
                until("Observed call · model.src")
                until("Source not recorded")
                pending.clear()
                send(b"b")
                until("Observed call · model.identity")
                pending.clear()
                send(b"b")
                until("2/2 ←→ model.identity")
                pending.clear()
                send(b"\x1b[D")
                until("1/2 ←→ model.src")
            if mode in ("fanout", "fanout_mono"):
                until("1/2 ←→ model.src")
                pending.clear()
                send(b"\n")
                until("Observed call · model.src")
                pending.clear()
                send(b"c")
                until("Later input matches · partial")
                until("34 recorded matches")
                pending.clear()
                send(b"g")
                until(">   1/34 model.identity")
                pending.clear()
                send(b"g")
                until(">├╌ 1/34 model.identity")
                pending.clear()
                send(b"\x1b[6~")
                until("33/34 model.branches.31")
                pending.clear()
                send(b"\x1b[<0;5;18M\x1b[<32;5;17M\x1b[<0;5;17m")
                send(b"g")
                until(">   19/34 model.branches.17")
                pending.clear()
                send(b"g")
                until(">├╌ 19/34 model.branches.17")
                pending.clear()
                send(b"\x1b[<0;5;18M\x1b[<0;5;18m")
                until(">├╌ 33/34 model.branches.31")
                pending.clear()
                send(b"F")
                until("model.branches.31   ·   Linear")
                until("? 1 matches from model.src")
                pending.clear()
                send(b"o")
                until("o show matches")
                pending.clear()
                send(b"o")
                until("o hide matches")
                pending.clear()
                send(b"z")
                until("2D")
                until("model.branches.31   ·   Linear")
                until("? 1 matches from model.src")
                until("origin not shown")
                pending.clear()
                send(b"\x1b[5~")
                previous_page = until("? 1 matches from model.src")
                assert b"model.branches.31   \xc2\xb7   Linear" not in previous_page
                pending.clear()
                send(b"F")
                until(">├╌ 33/34 model.branches.31")
                pending.clear()
                send(b"F")
                until("model.branches.31   ·   Linear")
                until("? 1 matches from model.src")
                pending.clear()
                send(b"z")
                until("model.branches.31   ·   Linear")
                pending.clear()
                send(b"x")
                until("x1/2 · no links · ? partial")
                until("model.branches.31   ·   Linear")
                until("? 1 matches from model.src")
                until("origin not shown")
                pending.clear()
                send(b"F")
                until(">├╌ 33/34 model.branches.31")
                pending.clear()
                send(b"\n")
                until("Observed call · model.branches.31")
                pending.clear()
                send(b"F")
                until("model.branches.31   ·   Linear")
                pending.clear()
                send(b"F")
                until("Observed call · model.branches.31")
                pending.clear()
                send(b"b")
                until("33/34 model.branches.31")
                pending.clear()
                send(b"c")
                until("Observed call · model.src")
                pending.clear()
                send(b"b")
                until("1/2 ←→ model.src")
            if mode == "shared":
                until("1/2 ←→ alias? model.src")
                pending.clear()
                send(b"\n")
                until("Shared call · registration unknown")
                pending.clear()
                send(b"F")
                until("Call location unknown")
                pending.clear()
                send(b"F")
                until("Shared call · registration unknown")
                pending.clear()
                send(b"b")
                until("1/2 ←→ alias? model.src")
            send(b"Fq")
            until("Model-Maven Workbench")
            send(b"q")
            child.wait(timeout=10)
            assert child.returncode == 0
            print(f"{width}x24 {graphics} {mode}: partial identity evidence verified")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
            os.close(master)
