#!/usr/bin/env python3
"""Replay long port-pair evidence against a real local m-model and PyTorch."""

import fcntl
import json
import os
import pty
import re
import select
import struct
import subprocess
import tempfile
import termios
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
scratch = tempfile.TemporaryDirectory(prefix="maven-pair-evidence-")
project = Path(scratch.name)
(project / "fixture.py").write_text(
    """from collections import OrderedDict
from torch import nn
class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self, '编码器segment_' * 8 + 'source', nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)))
    def forward(self, x):
        return next(self.children())(x)
def model(): return Wrapper()
def named():
    return nn.Sequential(OrderedDict([
        ('z_first', nn.Linear(8, 8)),
        ('a_second', nn.Linear(8, 8)),
        ('m_third', nn.Linear(8, 8)),
    ]))
class Sequential(nn.Sequential):
    def forward(self, value): return value
def custom(): return Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
def shared():
    layer = nn.Linear(8, 8)
    return nn.Sequential(OrderedDict([
        ('first', layer), ('middle', nn.Linear(8, 7)), ('again', layer),
    ]))
def batch():
    import torch
    return torch.ones(2, 8)
""",
    encoding="utf-8",
)
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get(
    "MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model")
)
custom_report = subprocess.run(
    [
        cli,
        "--json",
        "--no-interactive",
        "model",
        "inspect",
        "--model",
        "fixture:custom",
    ],
    cwd=project,
    capture_output=True,
    text=True,
    check=True,
)
custom_root = json.loads(custom_report.stdout)["data"]["modules"][0]
assert custom_root["type"] == "Sequential" and "sequential_children" in custom_root
assert custom_root["sequential_children"] is None
print("real CLI: custom class named Sequential carries no declared order")
name = "编码器segment_" * 8 + "source"
for mode in ("cells", "pixels"):
    master, slave = pty.openpty()

    def resize(rows, cols):
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    resize(24, 40)
    env = os.environ.copy()
    env.update(
        TERM="xterm-ghostty",
        NO_COLOR="1",
        MODEL_MAVEN_REDUCE_MOTION="1",
        MODEL_MAVEN_GRAPHICS=mode,
        MODEL_MAVEN_TUI_PREFERENCES=str(project / ("prefs-" + mode + ".json")),
    )
    env.pop("COLUMNS", None)
    env.pop("TMUX", None)
    child = subprocess.Popen(
        [tui, "--lang", "en", "--cli", cli],
        cwd=project,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    pending = bytearray()

    def until(token, timeout=30, frame=False):
        token = token.encode()
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            pos = pending.find(token)
            if pos >= 0:
                stop = pos + len(token)
                if frame:
                    n = pending.find(b"\x1b[?2026l", stop)
                    if n < 0:
                        stop = None
                    else:
                        stop = n + len(b"\x1b[?2026l")
                if stop is not None:
                    out = bytes(pending[:stop])
                    del pending[:stop]
                    return out.decode(errors="replace")
            if child.poll() is not None:
                raise RuntimeError(
                    f"child exited {child.returncode}: {pending[-2000:]!r}"
                )
            if select.select([master], [], [], 0.1)[0]:
                try:
                    pending.extend(os.read(master, 65536))
                except OSError:
                    pass
        raise RuntimeError(f"{mode}: missing {token!r}: {pending[-4000:]!r}")

    def send(v):
        os.write(master, v)

    try:
        until("Open workbench")
        send(b"\n1fixture:model\n")
        until("Completed", 60)
        send(b"b6")
        until("INSPECT · reported modules", 30, True)
        send(b"\x1b[B\n")
        until("Sequential · declared flow", 30, True)
        send(b"M")
        until("Port matrix", 30, True)
        send(b"\nM")
        until("Port pair · Connection unknown", 30, True)
        send(b"v")
        evidence = until("Port pair evidence", 30, True)
        normalized = (
            re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", evidence)
            .replace("\r", "")
            .replace("\n", "")
        )
        for suffix in (".out[0]", ".in[0]"):
            assert "model." + name + ".0" + suffix in normalized, (
                f"{mode}: missing complete {suffix}: {normalized}"
            )
        resize(14, 34)
        until("Port pair evidence", 30, True)
        send(b"\x1b[6~")
        tail = until("Port pair evidence", 30, True)
        assert "source.0.in[0]" in tail, (
            f"{mode}: input end missing after scroll: {tail}"
        )
        resize(30, 80)
        until("Port pair evidence", 30, True)
        send(b"v")
        until("Port pair · Connection unknown", 30, True)
        send(b"q")
        until("Model-Maven Workbench", 30)
        send(b"1\x15fixture:named\n")
        until("Completed", 60)
        send(b"b6")
        until("Sequential · declared flow", 30, True)
        send(b"w")
        until("model.z_first[0] → model.a_second[0]", 30, True)
        send(b"w")
        until("model.a_second[0] → model.m_third[0]", 30, True)
        send(b"q")
        until("Model-Maven Workbench", 30)
        send(b"1\x15fixture:shared\n")
        until("Completed", 60)
        send(b"b6")
        until("Sequential · declared flow", 30, True)
        send(b"w")
        until("model.first[0] → model.middle[0]", 30, True)
        send(b"w")
        until("model.middle[0] → model.again[0]", 30, True)
        send(b"/model.again")
        until("model.again", 30, True)
        send(b"\n")
        until("Shared instance · CLI module-2", 30, True)
        send(b"q")
        until("Model-Maven Workbench", 30)
        resize(24, 40)
        until("Model-Maven Workbench", 30)
        send(b"2\nfixture:batch\n")
        until("Shared instance · invoked", 60)
        send(b"b8")
        until("Shared call · path unknown", 30, True)
        send(b"F")
        until("Shared call · registration unknown", 30, True)
        send(b"Fq")
        until("Model-Maven Workbench", 30)
        send(b"q")
        child.wait(timeout=10)
        assert child.returncode == 0, child.returncode
        print(
            f"{mode}: real CLI inspect, full Unicode paths, named/shared Sequential order and ambiguous shared-call warnings verified"
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        os.close(master)
