#!/usr/bin/env python3
"""Exercise the language picker and saved welcome locale in a real PTY."""
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

tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(Path(__file__).resolve().parent.parent / "target/debug/m-model-tui"))
for width in (40, 80):
    with tempfile.TemporaryDirectory(prefix="maven-language-") as directory:
        project = Path(directory)
        preferences = project / "preferences.json"
        env = dict(os.environ, LC_ALL="C", TERM="xterm-256color", NO_COLOR="1",
                   MODEL_MAVEN_TUI_PREFERENCES=str(preferences), MODEL_MAVEN_REDUCE_MOTION="1")
        def launch():
            master, slave = pty.openpty()
            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))
            child = subprocess.Popen([tui, "--cli", "/no-such-model-maven-cli", "--project", str(project)],
                                     stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
            os.close(slave)
            return child, master
        def until(child, master, expected):
            data = bytearray()
            deadline = time.monotonic() + 30
            needle = expected.encode()
            while needle not in data:
                if time.monotonic() > deadline or child.poll() is not None:
                    raise AssertionError((expected, data[-6000:].decode(errors="replace")))
                if select.select([master], [], [], 0.2)[0]:
                    data.extend(os.read(master, 65536))
            return data.decode(errors="replace")
        def wait_and_drain(child, master):
            # The terminal output buffer can fill during repaint, especially on macOS.
            deadline = time.monotonic() + 20
            while child.poll() is None and time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        os.read(master, 65536)
                    except OSError:
                        break
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        child, master = launch()
        try:
            until(child, master, "Open workbench")
            os.write(master, b"\r")
            until(child, master, "Settings")
            os.write(master, b"s")
            until(child, master, "5. Language")
            os.write(master, b"5")
            until(child, master, "简体中文")
            os.write(master, b"2")
            until(child, master, "语言")
            os.write(master, b"b")
            until(child, master, "设置")
            os.write(master, b"q")
            wait_and_drain(child, master)
            assert child.returncode == 0
            assert json.loads(preferences.read_text())["language"] == "zh-cn"
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            os.close(master)
        child, master = launch()
        try:
            until(child, master, "进入工作台")
            os.write(master, b"q")
            wait_and_drain(child, master)
            assert child.returncode == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            os.close(master)
        print(f"{width} columns: full-screen picker switches immediately and survives restart")
