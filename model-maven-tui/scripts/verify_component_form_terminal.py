#!/usr/bin/env python3
"""Exercise component Schema prompts and cancellation in a full-screen PTY."""
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
with tempfile.TemporaryDirectory(prefix="maven-component-form-terminal-") as directory:
    project = Path(directory)
    (project / "fixture.py").write_text('''import torch
from torch import nn
def model(): return nn.Sequential(nn.Linear(4, 4), nn.ReLU())
def batch(): return torch.ones(2, 4)
def layer(negative_slope=0.1, tag="small"): return nn.LeakyReLU(negative_slope)
''')
    env = os.environ.copy()
    env.update(TERM="xterm-256color", NO_COLOR="1", MODEL_MAVEN_REDUCE_MOTION="1", MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    env.pop("TMUX", None)
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env, capture_output=True, text=True, timeout=60, check=True)
        return json.loads(result.stdout)
    run("init", "--model", "fixture:model", "--batch", "fixture:batch")
    schema = {"type":"object", "properties":{"negative_slope":{"type":"number", "minimum":0, "maximum":1, "default":0.1}, "tag":{"type":"string", "enum":["small", "big"]}}, "required":["tag"], "additionalProperties":False}
    run("component", "add", "demo/activation", "--kind", "layer", "--version", "1.2.3", "--entrypoint", "fixture:layer", "--params-schema", json.dumps(schema))
    modules = run("layer", "list", "--model", "fixture:model")["data"]["modules"]
    index = next(i for i, item in enumerate(modules, 1) if item["path"] == "model.1")
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
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
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-4000:]!r}")
            if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-4000:]!r}")
    def send(value): os.write(master, value)
    try:
        until("Open workbench")
        send(b"\nw")
        until("Model layers")
        send(b"b")
        until("Layer number")
        send(f"{index}\n".encode())
        until("Action: add")
        send(b"replace\n")
        until("Layer type")
        if os.environ.get("MODEL_MAVEN_FORM_BUILTIN") == "1":
            send(b"Dropout\n")
            until("Layer parameters as JSON object")
            send(b"\x15:form\n")
            until("Built-in Layer fields")
            send(b"b")
            until("Value for p")
            send(b"1.5\n")
            until("above the maximum")
            send(b"b")
            until("Value for p")
            send(b"0.25\n")
            until("Value for inplace")
            send(b"\n")
        else:
            send(b"component:demo/activation\n")
            until("Component parameter Schema")
            until("demo/activation@1.2.3")
            send(b"b")
            until("Parameters: fields or json")
            send(b"\n")
            until("Value for negative_slope")
            send(b"2\n")
            until("above the maximum")
            send(b"b")
            until("Value for negative_slope")
            send(b"0.25\n")
            until("Value for tag")
            send(b"small\n")
        until("Layer preview")
        send(b"b")
        until("Save this variant edit?")
        send(b"\n")
        until("Model-Maven Workbench")
        assert not (project / ".model-maven/variants/working.json").exists()
        send(b"q")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("Full-screen parameter form rejects an out-of-range field and safely cancels the preview")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
