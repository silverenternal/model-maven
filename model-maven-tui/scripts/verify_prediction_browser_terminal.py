#!/usr/bin/env python3
"""Browse real saved evaluation previews in two full-screen PTY sizes."""
import fcntl
import json
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
with tempfile.TemporaryDirectory(prefix="maven-predictions-pty-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), TERM="xterm-256color", NO_COLOR="1")
    (project / "samples.csv").write_text("x,y\n" + "".join(f"{i},{i+1}\n" for i in range(12)))
    (project / "evaluation.py").write_text('''import csv
from pathlib import Path
import torch
def data():
    with Path('samples.csv').open(newline='') as stream: return list(csv.DictReader(stream))
def evaluate(model, data, context):
    context.report_samples(len(data))
    for index, row in enumerate(data):
        with torch.no_grad():
            output = model(x=torch.tensor([[float(row['x']), float(row['y'])]]))
        context.report_prediction(f"sample-{index}", output[0].tolist(), target=[int(row['y'])])
    return {"metrics": {"count": len(data)}}
''')
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout); assert result.returncode == 0, report
        return report
    args = ("scaffold", "--input", "x=2", "--output-dim", "1")
    plan = run(*args); run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    trained = run("train", "--step", "model_starter:step", "--steps", "1", "--checkpoint-every", "1")
    train_id = trained["data"]["run_id"]
    evaluated = run("evaluate", "--run", train_id, "--evaluator", "evaluation:evaluate", "--data", "evaluation:data",
                    "--samples", "12", "--preview-samples", "12", "--data-fingerprint", "csv-eval-v1")
    evaluation_id = evaluated["data"]["run_id"]
    listing = run("runs", "list")["data"]["runs"]
    index = next(i for i, item in enumerate(listing, 1) if item["id"] == evaluation_id)
    saved = run("runs", "show", evaluation_id)["data"]["report"]["data"]
    assert saved["source_run_id"] == train_id and saved["provenance"]["data_identity"]["data_fingerprint"] == "csv-eval-v1"
    assert saved["checkpoint"]["sha256"] == trained["data"]["checkpoint"]["sha256"]
    assert saved["evaluation"]["sample_previews"]["samples"][0]["prediction"] != saved["evaluation"]["sample_previews"]["samples"][1]["prediction"]
    for width in (40, 100):
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))
        child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project, env=env,
                                 stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave); pending = bytearray()
        def until(token, timeout=35):
            needle = token.encode(); deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                position = pending.find(needle)
                if position >= 0:
                    end = position + len(needle); result = bytes(pending[:end]); del pending[:end]; return result
                if child.poll() is not None: raise AssertionError(f"TUI exited {child.returncode}: {pending[-4000:]!r}")
                if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
            raise AssertionError(f"missing {token!r}: {pending[-4000:]!r}")
        try:
            until("Open workbench"); os.write(master, b"\n")
            until("Model-Maven Workbench"); os.write(master, b"j")
            until(evaluation_id); os.write(master, b"b")
            until("Run number"); os.write(master, f"{index}\n".encode())
            until("Saved evaluation predictions"); first_page = until("sample-0")
            assert b"Declared data" in first_page and b"csv-eval-v1" in first_page, first_page
            pending.clear(); os.write(master, b"n")
            until("sample-8"); pending.clear(); os.write(master, b"3")
            until("Prediction detail"); until("sample-10"); until("Target: [11]")
            pending.clear(); os.write(master, b"b")
            until("> 3. sample-10"); pending.clear(); os.write(master, b"p")
            until("sample-0"); pending.clear(); os.write(master, b"i")
            until("Evaluation provenance"); until("Declared data version"); until("csv-eval-v1")
            print(f"{width}x24: page navigation, sample detail, return focus and saved provenance", flush=True)
        finally:
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=10)
            os.close(master)
