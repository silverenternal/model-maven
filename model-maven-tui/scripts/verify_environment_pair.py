#!/usr/bin/env python3
"""Recover a missing library through real, explicitly confirmed environment operations."""
import fcntl
import importlib.util
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
with tempfile.TemporaryDirectory(prefix="maven-env-pair-") as directory:
    project = Path(directory)
    wheelhouse = project / "wheelhouse"
    wheelhouse.mkdir()
    spec = importlib.util.spec_from_file_location("environment_fixture", root.parent / "model-maven/scripts/verify_environment_setup.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.build_wheel(wheelhouse, "fixture-dependency", "1.0.0", "fixture_dependency", "READY = True\n")
    (project / "requirements.txt").write_text("fixture-dependency==1.0.0\n")
    (project / "fixture.py").write_text("import torch\nfrom torch import nn\ndef model(): return nn.Linear(4, 2)\ndef batch(): return torch.ones(2, 4)\n")
    subprocess.run([cli, "--json", "--no-interactive", "init", "--model", "fixture:model", "--batch", "fixture:batch"], cwd=project, check=True, capture_output=True)
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    env = os.environ.copy()
    env.update(MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), PIP_NO_INDEX="1", PIP_FIND_LINKS=str(wheelhouse), PIP_DISABLE_PIP_VERSION_CHECK="1")
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    pending = bytearray()
    menu = "check, libraries, probe, configure, setup, refresh"
    def until(token, timeout=90):
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
    def finished(number, operation):
        for _ in range(180):
            send("refresh\n")
            page = until(menu)
            if f"{number}. {operation} · finished" in page:
                section = page.split(f"{number}. {operation} · finished", 1)[1]
                assert "Completed" in section, page
                return section
            time.sleep(0.5)
        raise AssertionError(f"{operation} did not finish: {page}")
    def setup(path, update, apply):
        send("setup\n")
        until("Requirements file")
        send("\n")
        until("Environment path")
        send(path + "\n")
        until("Re-resolve")
        send(("y" if update else "n") + "\n")
        until("Type apply to continue")
        send(("apply" if apply else "n") + "\n")
        return until(menu)
    def probe():
        send("probe\n")
        until("Library ID to import")
        send("torch\n")
        until("Run supported CPU smoke")
        send("n\n")
        return until(menu)
    try:
        until("Open workbench")
        send("\ny\n")
        until(menu)
        setup(".venv-isolated", False, False)
        assert not (project / ".venv-isolated").exists()
        setup(".venv-isolated", False, True)
        finished(1, "environment_setup")
        assert (project / ".venv-isolated").is_dir()
        probe()
        missing = finished(2, "environment_library_probe")
        assert '"not_installed"' in missing, missing
        send("configure\n")
        until("New environment policy")
        send("inherit\n")
        until("New environment path")
        send(".venv-shared\n")
        until("Type apply to continue")
        send("apply\n")
        until(menu)
        finished(3, "environment_configure")
        setup(".venv-shared", True, True)
        finished(4, "environment_setup")
        probe()
        restored = finished(5, "environment_library_probe")
        assert '"imported"' in restored, restored
        send("check\n")
        until(menu)
        finished(6, "environment_check")
        config = json.loads((project / ".model-maven/environment.json").read_text())
        assert config["venv"] == ".venv-shared" and config["system_site_packages"] is True, config
        send("b\n")
        until("Model-Maven Workbench")
        send("q\n")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("Explicit environment setup cancellation, missing Torch probe, inherited environment repair and successful retry verified")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
