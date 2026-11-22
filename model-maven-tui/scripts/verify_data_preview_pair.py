#!/usr/bin/env python3
"""Exercise background preview, navigation and cancellation with real CLI/PyTorch."""
import fcntl
import json
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
with tempfile.TemporaryDirectory(prefix="maven-data-preview-") as directory:
    project = Path(directory)
    (project / "fixture.py").write_text('''import time
import torch
from torch import nn
def model(): return nn.Linear(4, 2)
def batch(): return (torch.ones(2, 4), torch.tensor([0, 1]))
def slow():
    time.sleep(30)
    return batch()
def medium():
    time.sleep(8)
    return batch()
''')
    subprocess.run([cli, "--json", "--no-interactive", "init", "--model", "fixture:model", "--batch", "fixture:batch"], cwd=project, check=True, capture_output=True)
    other = project / "other"
    other.mkdir()
    (other / "fixture.py").write_text((project / "fixture.py").read_text())
    subprocess.run([cli, "--json", "--no-interactive", "init", "--model", "fixture:model", "--batch", "fixture:batch"], cwd=other, check=True, capture_output=True)
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "prefs.json")
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    pending = bytearray()
    def until(token, timeout=45):
        needle, deadline = token.encode(), time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos = pending.find(needle)
            if pos >= 0:
                end = pos + len(needle)
                result = bytes(pending[:end])
                del pending[:end]
                return result.decode(errors="replace")
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-3000:]!r}")
            if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-3000:]!r}")
    def send(data): os.write(master, data.encode())
    def start(factory):
        send("start\n")
        until("Source: batch or dataset")
        send("\n")
        until("Factory module:callable")
        send(factory + "\n")
        until("Split")
        send("\n")
        until("Maximum samples")
        send("\n")
        until("Factory parameters")
        send("\n")
        until("Renderer: auto, summary, or custom")
        send("summary\n")
        return until("start, refresh, cancel NUMBER")
    def refresh_until(text):
        for _ in range(25):
            send("refresh\n")
            page = until("start, refresh, cancel NUMBER")
            if text in page: return page
            time.sleep(0.5)
        raise AssertionError(f"preview never showed {text!r}: {page}")
    try:
        until("Open workbench")
        send("\nx\n")
        until("start, refresh, cancel NUMBER")
        page = start("fixture:batch")
        assert "running" in page, page
        send("b\n")
        until("Model-Maven Workbench")
        send("x\n")
        until("start, refresh, cancel NUMBER")
        page = refresh_until('"published":true')
        assert '"sample_count"' in page, page
        page = start("fixture:slow")
        assert "running" in page, page
        send("b\n")
        until("Model-Maven Workbench")
        send("o\n")
        until("Project directory")
        send(str(other) + "\n")
        until("Model-Maven Workbench")
        send("x\n")
        page = until("start, refresh, cancel NUMBER")
        assert "1. finished" not in page and "2. running" not in page and "Preview:" not in page, page
        send("b\n")
        until("Model-Maven Workbench")
        send("o\n")
        until("Project directory")
        send(str(project) + "\n")
        until("Model-Maven Workbench")
        send("x\n")
        page = until("start, refresh, cancel NUMBER")
        assert "1. finished" in page and "2. running" in page, page
        send("cancel 2\n")
        page = until("start, refresh, cancel NUMBER")
        assert "cancellation requested" in page or "finished" in page, page
        page = refresh_until("2. finished")
        assert '"published":true' in page  # prior successful receipt remains visible
        assert '"published":true' not in page.split("2. finished", 1)[1], page
        state_path = project / ".model-maven/previews/current.json"
        previous_generation = json.loads(state_path.read_text())["generation"]
        start("fixture:medium")
        for _ in range(100):
            if json.loads(state_path.read_text())["generation"] > previous_generation: break
            time.sleep(0.05)
        else: raise AssertionError("older preview never acquired a generation")
        start("fixture:batch")
        page = refresh_until("4. finished")
        assert '"published":true' in page.split("4. finished", 1)[1], page
        page = refresh_until("3. finished")
        older = page.split("3. finished", 1)[1].split("4. finished", 1)[0]
        assert '"published":false' in older and '"superseded"' in older, page
        send("b\n")
        until("Model-Maven Workbench")
        send("q\n")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("Background preview succeeds across navigation, isolates projects, cancels, and prevents older generations from publishing")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
