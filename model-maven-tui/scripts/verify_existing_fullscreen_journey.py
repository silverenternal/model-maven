#!/usr/bin/env python3
"""Connect, diagnose, edit and reopen a real existing model in a full-screen PTY."""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import shutil
import struct
import subprocess
import tempfile
import termios
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-existing-fullscreen-") as directory:
    project = Path(directory)
    source = project / "fixture.py"
    source.write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(in_features=32, out_features=8)
    def forward(self, x): return self.head(x)
def model(): return Model()
def batch(): return torch.ones(2, 48)
''')
    (project / "unrelated.txt").write_text("user-owned content\n")
    initial = source.read_text()
    env = dict(os.environ, TERM="xterm-256color", NO_COLOR="1", MODEL_MAVEN_REDUCE_MOTION="1",
               MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    env.pop("COLUMNS", None)
    env.pop("TMUX", None)
    transcript = bytearray()
    def launch():
        master, slave = pty.openpty()
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 28, 100, 0, 0))
        child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project, env=env,
                                 stdin=slave, stdout=slave, stderr=slave, close_fds=True)
        os.close(slave)
        return master, child, bytearray()
    def until(master, child, pending, token, timeout=30):
        needle = token.encode(); deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos = pending.find(needle)
            if pos >= 0:
                end = pos + len(needle); del pending[:end]; return
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-4000:]!r}")
            if select.select([master], [], [], 0.1)[0]:
                chunk = os.read(master, 65536); pending.extend(chunk); transcript.extend(chunk)
        raise AssertionError(f"missing {token!r}: {pending[-4000:]!r}")
    master, child, pending = launch()
    def wait(token): until(master, child, pending, token)
    def send(value): os.write(master, value)
    try:
        wait("Open workbench"); send(b"\nn")
        wait("Model factory (or :preset"); send(b"fixture:model\n")
        wait("Batch factory (optional)"); send(b"fixture:batch\n")
        wait("Project setup preview"); send(b"b")
        wait("Save project defaults?"); send(b"y")
        wait("Completed"); send(b"b")
        wait("Model-Maven Workbench"); send(b"1")
        wait("Model factory"); send(b"\n")
        wait("Completed"); send(b"b2\n")
        wait("Batch factory"); send(b"\n")
        wait("FORWARD_FAILED"); send(b"b8")
        wait("model.head   ·   Linear"); send(b"s")
        wait("Edit project Python source"); wait("Graph selection"); send(b"b")
        wait("add, set, replace"); send(b"set\n")
        wait("Source layer targets"); send(b"b")
        wait("Graph selection in current source model"); wait("inspected separately"); send(b"b")
        wait("Continue? Type y"); send(b"y\n")
        wait("Editable layer parameters"); send(b"b")
        wait("Layer parameters"); send(b'{"in_features":48}\n')
        wait("Source preview"); wait("fixture.py (candidate)"); send(b"b")
        wait("Type apply to modify source files"); send(b"apply\n")
        wait("Post-save verification"); wait('"status":"passed"'); send(b"b")
        wait("Model-Maven Workbench")
        assert "in_features=48" in source.read_text() and "in_features=32" in initial
        assert (project / "unrelated.txt").read_text() == "user-owned content\n"
        send(b"q"); child.wait(timeout=10); assert child.returncode == 0
    finally:
        if child.poll() is None: child.terminate(); child.wait(timeout=10)
        os.close(master)
    master, child, pending = launch()
    try:
        wait("Open workbench"); send(b"\n2\n")
        wait("Batch factory"); send(b"\n")
        wait("Completed")
        assert (project / ".model-maven/project.json").is_file()
        result = subprocess.run([cli, "--json", "--no-interactive", "model", "probe"], cwd=project,
                                env=env, capture_output=True, text=True, timeout=90, check=True)
        report = json.loads(result.stdout)
        assert report["status"] == "success" and any("model.head" in c["alias_paths"] for c in report["data"]["calls"]), report
        evidence = os.environ.get("MODEL_MAVEN_INTEGRATION_ARTIFACTS_DIR")
        if evidence:
            evidence = Path(evidence); evidence.mkdir(parents=True, exist_ok=False)
            (evidence / "terminal.ansi").write_bytes(transcript)
            (evidence / "before.py").write_text(initial)
            shutil.copy2(source, evidence / "after.py")
            (evidence / "reopened-probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            shutil.copy2(project / ".model-maven/project.json", evidence / "project.json")
        send(b"bq"); child.wait(timeout=10); assert child.returncode == 0
        print("Full-screen: connect existing project, diagnose failure, edit selected graph layer, restart and re-probe")
    finally:
        if child.poll() is None: child.terminate(); child.wait(timeout=10)
        os.close(master)
