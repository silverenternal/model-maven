#!/usr/bin/env python3
"""Replace a scaffold batch with a project CSV factory through the real TUI."""
import json
import fcntl
import pty
import select
import struct
import termios
import time
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-custom-data-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    (project / "examples.csv").write_text("1,2,3,4,0\n4,3,2,1,1\n")
    (project / "data_source.py").write_text('''import csv
import torch

def batch(path, scale, tag="default"):
    assert path == "examples.csv" and scale == 2 and tag in {"default", "override"}
    with open(path, newline="") as stream:
        rows = [[float(x) for x in row] for row in csv.reader(stream)]
    return {"args": [], "kwargs": {"features": torch.tensor([row[:4] for row in rows]) * scale},
            "loss_inputs": {"target": torch.tensor([[row[4], 1-row[4]] for row in rows])}}

batch.MODEL_MAVEN_STATELESS = True

def bad():
    return {"args": [], "kwargs": {"features": torch.ones(2, 3)}, "loss_inputs": {"target": torch.zeros(2, 2)}}
''')
    (project / "local_encoder.py").write_text("from torch import nn\ndef make(): return nn.ReLU()\n")
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=120)
        return result, json.loads(result.stdout)
    args = ("scaffold", "--input", "features=4", "--output-dim", "2")
    result, plan = run(*args)
    assert result.returncode == 0, plan
    result, report = run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    assert result.returncode == 0, report
    result, report = run("assembly", "add", "features_encoder", "--factory", "local_encoder:make")
    assert result.returncode == 0, report
    # Make the previous binding incompatible. Preview must explicitly use the
    # new binding instead of silently inheriting this saved value.
    result, binding_plan = run("config", "set-batch", "--batch", "model_starter:batch", "--batch-binding", "single")
    assert result.returncode == 0, binding_plan
    result, changed = run("config", "set-batch", "--batch", "model_starter:batch", "--batch-binding", "single", "--apply", "--if-current", binding_plan["data"]["from_revision"])
    assert result.returncode == 0, changed
    config_path = project / ".model-maven/project.json"
    original = config_path.read_bytes()
    def session(inputs, executable=cli):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", executable], cwd=project,
                              env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    bad = session("\na\nbatch\ndata_source:bad\nenvelope\n\nq\n")
    assert "FORWARD_FAILED" in bad.stdout, bad.stdout
    assert config_path.read_bytes() == original
    cancelled = session("\na\nbatch\ndata_source:batch\nenvelope\n{\"path\":\"examples.csv\",\"scale\":2}\nn\nq\n")
    assert "Batch change preview" in cancelled.stdout, cancelled.stdout
    assert config_path.read_bytes() == original
    # Change configuration only after the real forward and confirmation.
    wrapper = project / "race-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if "set-batch" in sys.argv and "--apply" in sys.argv:
    path = Path(".model-maven/project.json")
    config = json.loads(path.read_text())
    config["concurrent_marker"] = "keep"
    path.write_text(json.dumps(config))
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    race = session("\na\nbatch\ndata_source:batch\nenvelope\n{\"path\":\"examples.csv\",\"scale\":2}\ny\nq\n", str(wrapper))
    assert "PROJECT_CONFIG_CONFLICT" in race.stdout, race.stdout
    saved = json.loads(config_path.read_text())
    assert saved["concurrent_marker"] == "keep" and saved["batch"] == "model_starter:batch", saved
    accepted = session("\na\nbatch\ndata_source:batch\nenvelope\n{\"path\":\"examples.csv\",\"scale\":2}\ny\n"
                       "r\ntry-wait\n\n\n\n\n\n\ny\nq\n")
    assert "Batch change preview" in accepted.stdout and "Run started or completed" in accepted.stdout, accepted.stdout
    saved = json.loads(config_path.read_text())
    assert saved["batch"] == "data_source:batch" and saved["batch_binding"] == "envelope", saved
    assert saved["batch_params"] == {"path": "examples.csv", "scale": 2}, saved
    assert saved["model_injections"]["features_encoder"] == "local_encoder:make" and saved["concurrent_marker"] == "keep", saved
    result, history = run("config", "history")
    assert result.returncode == 0 and history["data"]["history"], history
    result, probe = run("model", "probe")
    assert result.returncode == 0, probe
    result, assembly = run("assembly", "show")
    assert result.returncode == 0 and assembly["data"]["validation"]["forward"] == "passed", assembly
    result, assembly_preview = run("assembly", "set", "features_encoder", "--factory", "local_encoder:make", "--dry-run")
    assert result.returncode == 0 and assembly_preview["data"]["validation"]["forward"] == "passed", assembly_preview
    result, generated = run("assembly", "generate", "--output", "generated/model_assembly.py")
    assert result.returncode == 0, generated
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 160, 0, 0))
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                             stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    pending = bytearray()
    def until(token, timeout=90):
        needle, deadline = token.encode(), time.monotonic() + timeout
        while time.monotonic() < deadline:
            index = pending.find(needle)
            if index >= 0:
                end = index + len(needle); page = bytes(pending[:end]); del pending[:end]
                return page.decode(errors="replace")
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-5000:]!r}")
            if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-5000:]!r}")
    def send(value): os.write(master, value.encode())
    try:
        until("Open workbench"); send("\nx\n")
        until("start, refresh, cancel NUMBER"); send("start\n")
        until("Source: batch or dataset"); send("\n")
        until("Factory module:callable"); send("\n")
        until("Split"); send("\n"); until("Maximum samples"); send("2\n")
        params_page = until("Factory parameters")
        assert '"path":"examples.csv"' in params_page and '"scale":2' in params_page, params_page
        send("\n"); until("Renderer: auto, summary, or custom"); send("summary\n")
        until("start, refresh, cancel NUMBER")
        for _ in range(100):
            send("refresh\n"); page = until("start, refresh, cancel NUMBER")
            if "1. finished" in page:
                assert '"published":true' in page.replace("\r", "").replace("\n", ""), page
                break
            time.sleep(0.2)
        else: raise AssertionError(f"preview did not finish: {page}")
        send("b\n"); until("Select:"); send("q\n"); child.wait(timeout=10)
        assert child.returncode == 0
    finally:
        if child.poll() is None: child.terminate(); child.wait(timeout=10)
        os.close(master)
    # Explicit keys overlay saved parameters only when the saved batch source is used.
    result, overridden = run("model", "probe", "--batch-param", 'tag="override"')
    assert result.returncode == 0, overridden
    result, other = run("model", "probe", "--batch", "data_source:bad", "--batch-binding", "envelope")
    assert result.returncode != 0 and other["diagnostics"][0]["code"] == "FORWARD_FAILED", other
    result, trained = run("train", "--step", "model_starter:step", "--steps", "1", "--checkpoint-every", "1")
    assert result.returncode == 0, trained
    train_id = trained["data"]["run_id"]
    result, snapshot = run("runs", "show", train_id)
    assert snapshot["data"]["inputs"]["batch_params"] == {"path": "examples.csv", "scale": 2}, snapshot
    result, new_plan = run("config", "set-batch", "--batch", "data_source:batch", "--batch-binding", "envelope", "--batch-param", 'path="examples.csv"', "--batch-param", "scale=3")
    assert result.returncode == 0, new_plan
    result, changed = run("config", "set-batch", "--batch", "data_source:batch", "--batch-binding", "envelope", "--batch-param", 'path="examples.csv"', "--batch-param", "scale=3", "--apply", "--if-current", new_plan["data"]["from_revision"])
    assert result.returncode == 0, changed
    result, resumed = run("train", "--resume", train_id, "--additional-steps", "1")
    assert result.returncode == 0, resumed
    result, resumed_snapshot = run("runs", "show", resumed["data"]["run_id"])
    assert resumed_snapshot["data"]["inputs"]["batch_params"] == {"path": "examples.csv", "scale": 2}, resumed_snapshot
    print("TUI validates and saves parameterized CSV batch; training resume retains its original parameters")
