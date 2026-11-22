#!/usr/bin/env python3
"""Preview a real Dataset as images and with a project-owned custom renderer."""
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
with tempfile.TemporaryDirectory(prefix="maven-dataset-preview-") as directory:
    project = Path(directory)
    (project / "fixture.py").write_text('''import torch
from torch import nn
def model(): return nn.Linear(4, 2)
def batch(): return torch.ones(2, 4)
class Dataset:
    def __len__(self): return 4
    def __getitem__(self, index): return torch.full((3,16,12), float(index)/4), index
    def sample_id(self, index): return f"sample-{index}"
def dataset(): return Dataset()
class HugeDataset:
    def __len__(self): return 1_000_000_000_000
    def __getitem__(self, index): return torch.tensor([index % 1000], dtype=torch.float32)
def huge(): return HugeDataset()
class Stream:
    def __iter__(self):
        for index in range(100): yield torch.tensor([index], dtype=torch.float32)
def stream(seed=0): return Stream()
def render(samples, output_directory, request):
    path = output_directory / "custom.html"
    path.write_text("<html><body>custom dataset preview</body></html>")
    return {"renderer_version": request["renderer_version"], "artifacts": [{"kind":"preview_document", "path":path.name, "media_type":"text/html"}]}
''')
    subprocess.run([cli, "--json", "--no-interactive", "init", "--model", "fixture:model", "--batch", "fixture:batch"], cwd=project, check=True, capture_output=True)
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 110, 0, 0))
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "prefs.json")
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
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
    def start(mode, factory="fixture:dataset", samples="2"):
        send("start\n")
        until("Source: batch or dataset")
        send("dataset\n")
        until("Factory module:callable")
        send(factory + "\n")
        until("Split")
        send("\n")
        until("Maximum samples")
        send(samples + "\n")
        until("Factory parameters")
        send("\n")
        until("Renderer: auto, summary, or custom")
        send(("" if mode == "auto" else mode) + "\n")
        if mode == "custom":
            until("Renderer factory")
            send("fixture:render\n")
            until("Renderer version")
            send("fixture-v1\n")
        until(menu)
    def finished(number):
        for _ in range(100):
            send("refresh\n")
            page = until(menu)
            if f"{number}. finished" in page:
                result = page.split(f"{number}. finished", 1)[1].replace("\r", "").replace("\n", "")
                assert '"published":true' in result and "Artifact:" in result, page
                if number == 3:
                    assert "seed 42" in result and "sha256-counter-rejection-v2" in result and "partial or unknown" in result, page
                return json.loads((project / ".model-maven/previews/current.json").read_text())["publication"]
            time.sleep(0.3)
        raise AssertionError(f"preview did not complete: {page}")
    try:
        until("Open workbench")
        send("\nx\n")
        until(menu)
        start("auto")
        report = finished(1)
        assert report["scope"]["source_kind"] == "dataset" and report["sample_count"] == 2 and report["rendered_image_count"] == 2, report
        artifacts = {item["kind"]: project / item["uri"] for item in report["artifacts"]}
        assert artifacts["image_grid"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert "data:image/png;base64," in artifacts["preview_document"].read_text()
        start("custom")
        report = finished(2)
        assert report["renderer"] == "project.custom" and report["renderer_version"] == "fixture-v1", report
        html = next(project / item["uri"] for item in report["artifacts"] if item["kind"] == "preview_document")
        assert "custom dataset preview" in html.read_text()
        start("summary", factory="fixture:huge", samples="2@42")
        huge_a = finished(3)
        assert huge_a["scope"]["seed"] == 42 and huge_a["scope"]["sampling_algorithm"] == "sha256-counter-rejection-v2", huge_a
        coverage_a = huge_a["scope"]["data_observation"]["coverage"]
        assert coverage_a["population_size"] == 1_000_000_000_000 and not coverage_a["complete"] and len(coverage_a["selected_indices"]) == 2, huge_a
        start("summary", factory="fixture:huge", samples="2@43")
        huge_b = finished(4)
        assert huge_b["scope"]["seed"] == 43 and huge_b["scope"]["data_observation"]["coverage"]["selected_indices"] != coverage_a["selected_indices"], huge_b
        start("summary", factory="fixture:stream", samples="3@42/20")
        stream = finished(5)
        scope = stream["scope"]
        coverage = scope["data_observation"]["coverage"]
        assert scope["sampling_algorithm"] == "sha256-window-rank-v1", stream
        assert coverage["kind"] == "bounded-window-sample" and coverage["examined_items"] == 20 and coverage["scan_limit"] == 20, stream
        assert len(set(coverage["selected_offsets"])) == 3 and all(0 <= n < 20 for n in coverage["selected_offsets"]), stream
        assert not coverage["complete"] and not coverage["representative"] and not coverage["deterministic_selection"], stream
        assert coverage["seed_forwarded_to_factory"] and coverage["deterministic_given_same_prefix"], stream
        print("TUI selects deterministic indexed subsets and bounded stream-window samples with honest coverage")
        send("b\n")
        until("Model-Maven Workbench")
        send("q\n")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("Dataset auto preview produces PNG and embedded HTML; versioned custom renderer produces its own document")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
