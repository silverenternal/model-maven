#!/usr/bin/env python3
"""Measure actual 1024-layer model diagnosis in full-screen PTYs on Linux."""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import statistics
import struct
import subprocess
import tempfile
import termios
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
for width, height, graphics in ((40, 24, "cells"), (100, 30, "pixels")):
    with tempfile.TemporaryDirectory(prefix="maven-large-real-pty-") as directory:
        project = Path(directory)
        (project / "fixture.py").write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Identity() for _ in range(1024)])
        self.head = nn.Linear(8, 2)
    def forward(self, x): return self.head(self.layers(x))
def model(): return Model()
def batch(): return torch.ones(2, 4)
''')
        env = dict(os.environ, TERM="xterm-256color", MODEL_MAVEN_GRAPHICS=graphics,
                   MODEL_MAVEN_REDUCE_MOTION="1", MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
        for key in ("COLUMNS", "TMUX", "NO_COLOR"): env.pop(key, None)
        report_start = time.monotonic()
        report_result = subprocess.run([cli, "--json", "--no-interactive", "model", "probe", "--model", "fixture:model", "--batch", "fixture:batch"],
                                       cwd=project, env=env, capture_output=True, timeout=120)
        report_seconds = time.monotonic() - report_start
        report = json.loads(report_result.stdout)
        assert report["status"] == "error" and report["data"]["trace_truncated"] is True
        assert len(report["data"]["calls"]) == 256
        inspected = subprocess.run([cli, "--json", "--no-interactive", "model", "inspect", "--model", "fixture:model"],
                                   cwd=project, env=env, capture_output=True, timeout=120, check=True)
        modules = json.loads(inspected.stdout)["data"]["modules"]
        assert len(modules) >= 1024, len(modules)
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project, env=env,
                                 stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        pending = bytearray()
        total = 0
        def until(token, timeout=30):
            global total
            needle = token.encode(); deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                position = pending.find(needle)
                if position >= 0:
                    end = position + len(needle)
                    frame = bytes(pending[:end]); del pending[:end]
                    return frame
                if child.poll() is not None: raise RuntimeError(f"TUI exited {child.returncode}: {pending[-4000:]!r}")
                if select.select([master], [], [], 0.1)[0]:
                    chunk = os.read(master, 65536); pending.extend(chunk); total += len(chunk)
            raise AssertionError(f"missing {token!r}: {pending[-4000:]!r}")
        def settle():
            global total
            deadline = time.monotonic() + 2
            pending.clear()
            while time.monotonic() < deadline:
                if not select.select([master], [], [], 0.05)[0]: return
                chunk = os.read(master, 65536)
                total += len(chunk)
            raise AssertionError("terminal did not settle before interaction timing")
        try:
            until("Open workbench")
            start = time.monotonic()
            os.write(master, b"\n1fixture:model\n")
            until("Completed")
            inspect_seconds = time.monotonic() - start
            os.write(master, b"b2\n")
            until("Batch factory")
            start = time.monotonic(); os.write(master, b"fixture:batch\n")
            until("FORWARD_FAILED"); probe_seconds = time.monotonic() - start
            settle(); start = time.monotonic(); os.write(master, b"b8")
            until("model.head   ·   Linear"); graph_seconds = time.monotonic() - start
            settle(); start = time.monotonic(); os.write(master, b"F")
            until("Failure path"); detail_seconds = time.monotonic() - start
            samples = []
            for _ in range(5):
                settle(); start = time.monotonic(); os.write(master, b"F")
                until("model.head   ·   Linear"); samples.append(time.monotonic() - start)
                settle(); start = time.monotonic(); os.write(master, b"F")
                until("Failure path"); samples.append(time.monotonic() - start)
            print(json.dumps({"terminal":f"{width}x{height}", "graphics":graphics,
                              "identity_layers":1024, "inspected_modules":len(modules), "inspect_report_bytes":len(inspected.stdout), "recorded_calls":256, "trace_truncated":True,
                              "probe_report_bytes":len(report_result.stdout), "probe_seconds":round(report_seconds,3),
                              "tui_inspect_seconds":round(inspect_seconds,3), "tui_probe_seconds":round(probe_seconds,3),
                              "graph_open_seconds":round(graph_seconds,3), "detail_open_seconds":round(detail_seconds,3),
                              "toggle_median_ms":round(statistics.median(samples)*1000,1),
                              "toggle_max_ms":round(max(samples)*1000,1), "pty_bytes_observed":total}), flush=True)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=10)
            os.close(master)
