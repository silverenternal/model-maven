#!/usr/bin/env python3
"""Reject a project configuration change during input capture."""
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
from pathlib import Path
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)
    def forward(self, x): return self.head(x)
def model(): return Model()
def batch():
    with Path("batch-calls.txt").open("a") as stream: stream.write("x")
    return torch.randn(2, 4)
''')
    original = source.read_bytes()
    subprocess.run([cli, "--json", "--no-interactive", "init", "--model", "fixture:model", "--batch", "fixture:batch"], cwd=project, check=True, capture_output=True)
    wrapper = project / "record-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
with Path("commands.jsonl").open("a") as stream: stream.write(json.dumps(sys.argv[1:]) + "\\n")
if "--capture-input-to" in sys.argv:
    path = Path(".model-maven/project.json")
    config = json.loads(path.read_text())
    config["concurrent_marker"] = "preserve-me"
    path.write_text(json.dumps(config))
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    env = os.environ.copy()
    env.update(TERM="xterm-256color", NO_COLOR="1", MODEL_MAVEN_REDUCE_MOTION="1", MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    env.pop("TMUX", None)
    child = subprocess.Popen([tui, "--lang", "en", "--cli", str(wrapper)], cwd=project, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
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
        send(b"set+snapshot\n")
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
        until("Capturing input snapshot")
        until("Temporary file")
        send(b"b")
        until("Project configuration changed during input capture")
        send(b"b")
        until("Temporary input snapshot removed")
        send(b"b")
        until("Model-Maven Workbench")
        import json
        commands = [json.loads(line) for line in (project / "commands.jsonl").read_text().splitlines()]
        captures = [args[args.index("--capture-input-to") + 1] for args in commands if "--capture-input-to" in args]
        assert len(captures) == 1 and not Path(captures[0]).parent.exists(), commands
        assert not any("--replay-input-from" in args for args in commands), commands
        assert json.loads((project / ".model-maven/project.json").read_text())["concurrent_marker"] == "preserve-me"
        assert source.read_bytes() == original
        send(b"q")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("TUI rejects changed project configuration during capture and removes the temporary input")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
