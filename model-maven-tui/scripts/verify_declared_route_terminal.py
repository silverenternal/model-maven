#!/usr/bin/env python3
"""Show a configured route alongside real failed-call evidence in a PTY."""
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
with tempfile.TemporaryDirectory(prefix="maven-declared-route-") as directory:
    project = Path(directory)
    factory = project / "components.py"
    factory.write_text("from torch import nn\ndef encoder(): return nn.Identity()\n")
    env = dict(os.environ, TERM="xterm-256color", NO_COLOR="1", MODEL_MAVEN_REDUCE_MOTION="1",
               MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    env.pop("COLUMNS", None)
    env.pop("TMUX", None)
    def run(*args, success=True):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        assert (proc.returncode == 0) == success, report
        return report
    args = ("scaffold", "--input", "a=4", "--input", "b=2", "--output-dim", "3", "--inject", "b_encoder=components:encoder")
    plan = run(*args)
    run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    factory.write_text("from torch import nn\ndef encoder(): return nn.Linear(5, 3)\n")
    failed = run("model", "probe", success=False)
    origin = failed["data"]["origin"]
    assert origin["scope"] == "partial_failure_context" and set(origin) == {"scope", "params", "injection_evidence", "declared_routes"}, origin
    assert origin["params"] == {}, origin
    declared = origin["declared_routes"]
    assert declared["scope"] == "model_self_reported_at_construction" and declared["status"] == "available", declared
    assert any(item == {"slot":"b_encoder", "input":"b", "registered_paths":["model.encoders.enc_b"]} for item in declared["routes"]), declared
    assert any(item["slot"] == "b_encoder" and item["registered_paths"] == ["model.encoders.enc_b"] for item in origin["injection_evidence"]), origin
    assert failed["data"]["failure_location"]["last_observed_module_call"]["alias_paths"] == ["model.encoders.enc_b"], failed
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 28, 100, 0, 0))
    child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project, stdin=slave,
                             stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    pending = bytearray()
    def until(token, timeout=60):
        needle = token.encode(); deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            position = pending.find(needle)
            if position >= 0:
                end = position + len(needle); del pending[:end]; return
            if child.poll() is not None: raise AssertionError(f"terminal exited: {pending[-4000:]!r}")
            if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-4000:]!r}")
    def send(value): os.write(master, value)
    try:
        until("Open workbench"); send(b"\n1")
        until("Model factory"); send(b"model_starter:model\n")
        until("Completed"); send(b"b2\n")
        until("Batch factory"); send(b"model_starter:batch\n")
        until("FORWARD_FAILED"); send(b"b8")
        until("model.encoders.enc_b"); send(b"F")
        until("Model-declared route · not observed flow"); until("b → b_encoder")
        send(b"F"); send(b"q")
        print("Failed forward retains model-declared default route; focused graph distinguishes it from observed flow")
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=10)
        os.close(master)
