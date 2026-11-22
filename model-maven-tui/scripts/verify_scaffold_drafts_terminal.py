#!/usr/bin/env python3
"""Full-screen PTY check of interrupted custom starter form drafts."""
import fcntl
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import tempfile
import termios
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-draft-pty-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), TERM="xterm-256color", NO_COLOR="1")
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project, env=env,
                             stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    captured = bytearray()
    def wait_for(marker, timeout=30):
        start = len(captured); deadline = time.monotonic() + timeout
        while marker.encode() not in captured[start:]:
            assert time.monotonic() < deadline, (marker, bytes(captured[-10000:]).decode(errors="replace"))
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready: captured.extend(os.read(master, 65536))
    try:
        wait_for("Open workbench"); os.write(master, b"\r")
        wait_for("Model-Maven Workbench"); os.write(master, b"m")
        wait_for("Inputs and optional routes"); os.write(master, b"a=4;b=2;@a_encoder=b\r")
        wait_for("Output features"); os.write(master, b"\x1b")
        wait_for("Model-Maven Workbench"); os.write(master, b"m")
        wait_for("a=4;b=2;@a_encoder=b"); os.write(master, b"\r")
        wait_for("Output features"); os.write(master, b"3\r")
        wait_for("Custom model preview"); os.write(master, b"b")
        wait_for("Create files?"); os.write(master, b"n\r")
        wait_for("Model-Maven Workbench"); os.write(master, b"m")
        wait_for("a=4;b=2;@a_encoder=b"); os.write(master, b"\r")
        wait_for("Output features"); os.write(master, b"\r")
        wait_for("Custom model preview"); os.write(master, b"b")
        wait_for("Create files?"); os.write(master, b"y\r")
        wait_for("Custom model project created.")
        assert '"a_encoder": "b"' in (project / "model_starter.py").read_text()
        print("Full-screen form retains inputs and output after leaving and cancelling the preview")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=10)
        os.close(master)
