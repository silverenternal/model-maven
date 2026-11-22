#!/usr/bin/env python3
"""Create, preview real image data, and try training in one TUI session."""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import tempfile
import termios
import time
from PIL import Image

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
preset = os.environ.get("MODEL_MAVEN_CLASSIFICATION_PRESET", "tiny-cnn")
if preset not in ("tiny-cnn", "resnet18"):
    raise ValueError("unsupported classification fixture preset")
if preset == "resnet18":
    try:
        import torchvision
        print(f"ResNet18 environment: torchvision={torchvision.__version__}", flush=True)
    except Exception as error:
        print(f"SKIPPED: ResNet18 needs usable torchvision: {type(error).__name__}: {error}", flush=True)
        sys.exit(77)
with tempfile.TemporaryDirectory(prefix="maven-classification-journey-") as directory:
    project = Path(directory)
    for name, color in [("cats", "red"), ("dogs", "blue")]:
        path = project / "images" / name
        path.mkdir(parents=True)
        Image.new("RGB", (20, 16), color).save(path / "example.png")
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 120, 0, 0))
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                             stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    pending = bytearray()
    transcript = bytearray()
    actions = []
    def until(token, timeout=90):
        needle, deadline = token.encode(), time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos = pending.find(needle)
            if pos >= 0:
                end = pos + len(needle)
                result = bytes(pending[:end]); del pending[:end]
                return result.decode(errors="replace")
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-5000:]!r}")
            if select.select([master], [], [], 0.1)[0]:
                chunk = os.read(master, 65536)
                pending.extend(chunk)
                transcript.extend(chunk)
        raise AssertionError(f"missing {token!r}: {pending[-5000:]!r}")
    def send(text):
        actions.append(text)
        os.write(master, text.encode())
    menu = "start, refresh, cancel NUMBER"
    try:
        until("Open workbench"); send("\nb\n")
        until("Preset: tiny-cnn"); send(("\n" if preset == "tiny-cnn" else "resnet18\n"))
        until("Image directory"); send("images\n")
        until("Create files? Type y"); send("y\n")
        until("Classification project created.")
        until("Select:"); send("x\n")
        until(menu); send("start\n")
        until("Source: batch or dataset"); send("\n")
        current = until("Factory module:callable")
        assert "classification:batch" in current and "suggested below" in current, current
        send("\n"); until("Split"); send("\n")
        until("Maximum samples"); send("2\n")
        until("Factory parameters"); send("\n")
        until("Renderer: auto, summary, or custom"); send("\n")
        until(menu)
        for _ in range(100):
            send("refresh\n"); page = until(menu)
            if "1. finished" in page:
                assert '"published":true' in page.replace("\r", "").replace("\n", ""), page
                break
            time.sleep(0.2)
        else: raise AssertionError(f"preview did not finish: {page}")
        publication = json.loads((project / ".model-maven/previews/current.json").read_text())["publication"]
        assert publication["sample_count"] == 2 and publication["rendered_image_count"] == 2, publication
        image_grid = next(project / item["uri"] for item in publication["artifacts"] if item["kind"] == "image_grid")
        assert image_grid.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        send("b\n"); until("Select:"); send("2\n")
        until("Model factory"); send("\n"); until("Batch factory"); send("\n")
        until("Observed calls for this batch"); until("Select:"); send("r\n")
        until("Run type"); send("try-wait\n"); until("Training mode"); send("\n")
        until("Training step"); send("\n"); until("Step/loop parameters"); send("\n")
        until("Number of steps"); send("\n"); until("Device:"); send("\n")
        until("Apply saved supervision"); send("\n")
        plan = until("Start this run? Type y")
        assert "classification:step" in plan and "classification:batch" in plan, plan
        send("y\n"); until("Run started or completed", timeout=120)
        until("Select:"); send("q\n"); child.wait(timeout=10)
        assert child.returncode == 0
        listing = subprocess.run([cli, "--json", "--no-interactive", "runs", "list"], cwd=project,
                                 env=env, capture_output=True, text=True, check=True, timeout=30)
        runs = json.loads(listing.stdout)["data"]["runs"]
        trial = next((item for item in runs if item["operation"] == "training_probe" and item["status"] == "completed"), None)
        assert trial is not None, runs
        evidence_path = os.environ.get("MODEL_MAVEN_INTEGRATION_ARTIFACTS_DIR")
        if evidence_path:
            evidence = Path(evidence_path)
            evidence.mkdir(parents=True, exist_ok=False)
            detail = subprocess.run([cli, "--json", "--no-interactive", "runs", "show", trial["id"]], cwd=project,
                                    env=env, capture_output=True, text=True, check=True, timeout=30)
            (evidence / "terminal-output.txt").write_bytes(transcript)
            (evidence / "terminal-input.json").write_text(json.dumps(actions, indent=2) + "\n")
            (evidence / "training-report.json").write_text(detail.stdout)
            (evidence / "classification.py").write_bytes((project / "classification.py").read_bytes())
            (evidence / "data-preview.json").write_text(json.dumps(publication, indent=2) + "\n")
        print(f"Real image classification ({preset}): TUI creates, previews saved batch, probes and runs one training step")
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=10)
        os.close(master)
