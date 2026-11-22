#!/usr/bin/env python3
"""Start, navigate, inspect and cancel real background trial workers."""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import tempfile
import termios
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-background-trial-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=90, check=True)
        return json.loads(result.stdout)
    args = ("scaffold", "--input", "image=3,4,4", "--input", "tabular=5", "--output-dim", "3")
    preview = run(*args)
    run(*args, "--apply", "--if-plan", preview["data"]["plan_revision"])
    source = project / "model_starter.py"
    source.write_text(source.read_text().replace("def step(model, batch, context):", "def step(model, batch, context):\n    import time, os\n    from pathlib import Path\n    Path('worker.pid').write_text(str(os.getpid()))\n    time.sleep(5)"))
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 140, 0, 0))
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                             stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    pending = bytearray()
    def until(token, timeout=90):
        needle, deadline = token.encode(), time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos = pending.find(needle)
            if pos >= 0:
                end = pos + len(needle)
                result = bytes(pending[:end]); del pending[:end]
                return result.decode(errors="replace")
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-5000:]!r}")
            if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-5000:]!r}")
    def send(value): os.write(master, value.encode())
    def launch():
        send("r\n"); until("Run type"); send("\n")
        until("Training mode"); send("\n"); until("Training step"); send("\n")
        until("Step/loop parameters"); send("\n"); until("Number of steps"); send("\n")
        until("Device:"); send("\n"); until("Apply saved supervision"); send("\n")
        until("Start this run? Type y"); send("y\n")
        until("Trial started in this session"); until("Select:")
    runs_prompt = "Run number, refresh"
    try:
        until("Open workbench"); send("\n"); until("Select:")
        launch()
        send("r\n"); until("Run type"); send("\n")
        until("A trial is already running"); until("Select:")
        send("j\n"); page = until(runs_prompt)
        assert "Trial task 1: running" in page, page
        send("b\n"); until("Select:")
        send("j\n"); until(runs_prompt)
        for _ in range(100):
            send("refresh\n"); page = until(runs_prompt)
            if "Trial task 1: finished" in page:
                assert "Completed" in page, page
                break
            time.sleep(0.2)
        else: raise AssertionError(f"trial did not finish: {page}")
        send("b\n"); until("Select:")
        previous_pid = int((project / "worker.pid").read_text())
        source.write_text(source.read_text().replace("time.sleep(5)", "time.sleep(30)"))
        launch()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            worker_pid = int((project / "worker.pid").read_text())
            if worker_pid != previous_pid:
                break
            time.sleep(0.1)
        else: raise AssertionError("second trial worker never entered its step")
        send("j\n"); until(runs_prompt); send("cancel trial 2\n")
        page = until(runs_prompt)
        assert "Trial task 2: cancellation requested" in page or "Trial task 2: finished" in page, page
        for _ in range(100):
            if "Trial task 2: finished" in page: break
            time.sleep(0.2); send("refresh\n"); page = until(runs_prompt)
        else: raise AssertionError(f"cancel did not terminate: {page}")
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(f"cancelled worker {worker_pid} is still alive")
        send("b\n"); until("Select:")
        previous_pid = worker_pid
        launch()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            worker_pid = int((project / "worker.pid").read_text())
            if worker_pid != previous_pid:
                break
            time.sleep(0.1)
        else: raise AssertionError("third trial worker never entered its step")
        send("q\n"); child.wait(timeout=10)
        assert child.returncode == 0
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(f"worker {worker_pid} survived TUI exit")
        print("Background trial supports navigation, duplicate guard, result refresh, cancellation and exit cleanup")
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=10)
        os.close(master)
