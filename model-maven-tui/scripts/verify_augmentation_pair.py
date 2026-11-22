#!/usr/bin/env python3
"""Exercise guarded augmentation editing and real transformed-data preview."""
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
with tempfile.TemporaryDirectory(prefix="maven-augmentation-pair-") as directory:
    project = Path(directory)
    (project / "fixture.py").write_text('''import torch
from torch import nn
def model(): return nn.Linear(4, 2)
def batch(): return torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4)
''')
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    run("init", "--model", "fixture:model", "--batch", "fixture:batch")
    wrapper = project / "cli-wrapper.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, subprocess, sys
real = {cli!r}
args = sys.argv[1:]
marker = {str(project / "inject")!r}
if "augment" in args and "--if-config-revision" in args and "--dry-run" not in args and os.path.exists(marker):
    os.unlink(marker)
    subprocess.run([real, "--json", "--no-interactive", "augment", "add", "RandomHorizontalFlip", "--id", "concurrent", "--target", "args.0", "--param", "p=0.25"], stdout=subprocess.DEVNULL, check=True)
os.execv(real, [real, *args])
''')
    wrapper.chmod(0o700)
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 110, 0, 0))
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "prefs.json")
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    pending = bytearray()
    menu = "start, refresh, cancel NUMBER"
    def until(token, timeout=60):
        needle, deadline = token.encode(), time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos = pending.find(needle)
            if pos >= 0:
                end = pos + len(needle)
                result = bytes(pending[:end])
                del pending[:end]
                return result.decode(errors="replace")
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-5000:]!r}")
            if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-5000:]!r}")
    def send(value): os.write(master, value.encode())
    def add(step, save, race=False):
        send("augment\n")
        until("Augmentation split")
        send("\n")
        until("add, set, move, remove")
        send("add\n")
        until("Type: Resize")
        send("RandomHorizontalFlip\n")
        until("Step ID")
        send(step + "\n")
        until("Batch target path")
        send("args.0\n")
        until("Transform parameters")
        send('{"p":1.0}\n')
        until("Augmentation preview")
        until("Save augmentation?")
        if race: (project / "inject").touch()
        send(("y" if save else "n") + "\n")
        return until(menu)
    def edit(action, number, params=None, before=None):
        send("augment\n")
        until("Augmentation split")
        send("\n")
        until("add, set, move, remove")
        send(action + "\n")
        until("Step number")
        send(str(number) + "\n")
        if action == "set":
            until("Batch target path")
            send("\n")
            until("Transform parameters")
            send(json.dumps(params) + "\n")
        if action == "move":
            until("Move before step ID")
            send((before or "") + "\n")
        until("Augmentation preview")
        until("Save augmentation?")
        send("y\n")
        page = until(menu)
        assert "Saved." in page, page
    try:
        until("Open workbench")
        send("\nx\n")
        until(menu)
        add("flip", False)
        assert run("augment", "list")["data"]["steps"] == []
        page = add("flip", True)
        assert "Saved." in page, page
        assert [step["id"] for step in run("augment", "list")["data"]["steps"]] == ["flip"]
        page = add("raced", True, race=True)
        assert "PROJECT_CONFIG_CONFLICT" in page, page
        assert [step["id"] for step in run("augment", "list")["data"]["steps"]] == ["flip", "concurrent"]
        send("start\n")
        until("Source: batch or dataset")
        send("\n")
        until("Factory module:callable")
        send("fixture:batch\n")
        until("Split")
        send("\n")
        until("Maximum samples")
        send("\n")
        until("Factory parameters")
        send("\n")
        until("Renderer: auto, summary, or custom")
        send("summary\n")
        until(menu)
        for _ in range(60):
            send("refresh\n")
            page = until(menu)
            if '"published":true' in page.replace("\r", "").replace("\n", ""): break
            time.sleep(0.3)
        else: raise AssertionError(f"data preview did not finish: {page}")
        normalized = page.replace("\r", "").replace("\n", "")
        assert '"transform_observations"' in normalized and "RandomHorizontalFlip" in normalized, page
        edit("set", 1, params={"p": 0.75})
        edit("move", 2, before="flip")
        steps = run("augment", "list")["data"]["steps"]
        assert [step["id"] for step in steps] == ["concurrent", "flip"] and steps[1]["parameters"]["p"] == 0.75
        edit("remove", 1)
        assert [step["id"] for step in run("augment", "list")["data"]["steps"]] == ["flip"]
        send("b\n")
        until("Model-Maven Workbench")
        send("q\n")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("Augmentation add/set/move/remove, cancellation, guarded conflict and actual transformed batch preview verified")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
