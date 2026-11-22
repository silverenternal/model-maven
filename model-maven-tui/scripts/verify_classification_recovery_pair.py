#!/usr/bin/env python3
"""Recover from a missing classification dependency and an unreadable local image."""
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
from PIL import Image

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-classification-recovery-") as directory:
    project = Path(directory)
    for name in ("cats", "dogs"):
        (project / "images" / name).mkdir(parents=True)
    corrupt = project / "images/cats/broken.png"
    corrupt.write_bytes(b"this is not a PNG")
    Image.new("RGB", (16, 16), "blue").save(project / "images/dogs/good.png")
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 140, 0, 0))
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
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
    try:
        until("Open workbench"); send("\n")
        until("Select:")
        torchvision_available = subprocess.run(["python3", "-c", "import torchvision"], capture_output=True).returncode == 0
        if not torchvision_available:
            send("b\n"); until("Preset: tiny-cnn"); send("resnet18\n")
            until("Image directory"); send("images\n")
            missing = until("Select:")
            assert "PRESET_DEPENDENCY_MISSING" in missing and "torchvision" in missing, missing
            assert not (project / "classification.py").exists() and not (project / ".model-maven/project.json").exists()
        else:
            print("torchvision is available; missing-dependency branch skipped")
        send("b\n"); until("Preset: tiny-cnn"); send("\n")
        until("Image directory"); send("images\n")
        until("Create files? Type y"); send("y\n")
        until("Classification project created."); until("Select:"); send("2\n")
        until("Model factory"); send("\n"); until("Batch factory"); send("\n")
        failed = until("Select:")
        assert "CLASS_SAMPLE_LOAD_FAILED" in failed and "cats/broken.png" in failed, failed
        source = (project / "classification.py").read_bytes()
        config = (project / ".model-maven/project.json").read_bytes()
        Image.new("RGB", (16, 16), "red").save(corrupt)
        send("2\n"); until("Model factory"); send("\n"); until("Batch factory"); send("\n")
        until("Observed calls for this batch")
        assert (project / "classification.py").read_bytes() == source
        assert (project / ".model-maven/project.json").read_bytes() == config
        until("Select:"); send("q\n"); child.wait(timeout=10)
        assert child.returncode == 0
        print("Repaired corrupt image probes successfully in the same TUI session")
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=10)
        os.close(master)
