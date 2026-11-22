#!/usr/bin/env python3
"""Select an inspected module in the real graph and hand it to variant editing."""
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
with tempfile.TemporaryDirectory(prefix="maven-graph-edit-") as directory:
    project = Path(directory)
    (project / "fixture.py").write_text('''import torch
from torch import nn
def model(): return nn.Sequential(nn.Linear(4, 4), nn.ReLU())
def batch(): return torch.ones(2, 4)
''')
    subprocess.run([cli, "--json", "--no-interactive", "init", "--model", "fixture:model", "--batch", "fixture:batch"], cwd=project, capture_output=True, check=True)
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    env = os.environ.copy()
    env.update(TERM="xterm-256color", NO_COLOR="1", MODEL_MAVEN_REDUCE_MOTION="1", MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"))
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
                return result
            if child.poll() is not None:
                raise AssertionError(f"terminal exited {child.returncode}: {pending[-4000:]!r}")
            if select.select([master], [], [], 0.1)[0]:
                pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-4000:]!r}")
    def send(data): os.write(master, data)
    try:
        until("Open workbench")
        send(b"\n1fixture:model\n")
        until("Completed")
        send(b"b6")
        until("E:edit")
        send(b"e")
        until("Model layers")
        send(b"b")
        until("Graph selection in current variant")
        selection = until("Graph model")
        assert b"model.0" in selection and b"Linear" in selection, selection
        until("Edit target model")
        send(b"b")
        until("Continue? Type y")
        send(b"n\n")
        until("Model-Maven Workbench")
        assert not (project / ".model-maven/variants/working.json").exists()
        send(b"6")
        until("E:edit")
        send(b"e")
        until("Model layers")
        send(b"b")
        until("Graph selection in current variant")
        until("Edit target model")
        send(b"b")
        until("Continue? Type y")
        send(b"y\n")
        until("Action: add")
        send(b"set\n")
        until("Editable layer parameters")
        until("bias = true")
        send(b"b")
        until("Layer parameters")
        send(b":list\n")
        until("Editable layer parameters")
        send(b"b")
        until("Layer parameters")
        send(b"missing_field\n")
        until("Unknown field")
        send(b"b")
        until("Layer parameters")
        send(b"bias\n")
        until("JSON value for bias")
        send(b"\x15not-json\n")
        until("Invalid JSON value")
        send(b"b")
        until("Layer parameters")
        send(b"bias\n")
        until("JSON value for bias")
        send(b"\x15false\n")
        until("Layer parameters")
        send(b"2\n")
        until("JSON value for in_features")
        send(b"\n")
        until("Layer parameters")
        send(b":done\n")
        until("Layer preview")
        until("Validation:")
        send(b"b")
        until("Save this variant edit?")
        send(b"y\n")
        until("independent recheck")
        until("Completed")
        send(b"b")
        until("Model-Maven Workbench")
        listing = subprocess.run([cli, "--json", "--no-interactive", "layer", "list", "--variant", "working"], cwd=project, capture_output=True, text=True, check=True)
        assert next(item for item in json.loads(listing.stdout)["data"]["modules"] if item["path"] == "model.0")["editable_params"]["bias"] is False
        send(b"q")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("Graph selection cancels safely, then edits a layer in the variant and independently rechecks it")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
