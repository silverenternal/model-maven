#!/usr/bin/env python3
"""Check actual shared/branch/clone/mutation evidence and truncation in TUI."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-graph-evidence-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="40")
    (project / "fixture.py").write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, long=False):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.alias = self.shared
        self.left = nn.Identity()
        self.right = nn.Identity()
        self.clone_sink = nn.Identity()
        self.inplace_sink = nn.Identity()
        self.unvisited = nn.ReLU()
        self.long = nn.Sequential(*[nn.Identity() for _ in range(1024)]) if long else nn.Identity()
        self.head = nn.Linear(8, 1)
    def forward(self, x):
        value = self.shared(x)
        self.left(value)
        self.clone_sink(value.clone())
        value.add_(1)
        self.inplace_sink(value)
        self.right(self.alias(x + 1))
        return self.head(self.long(value))
def short(): return Model()
def large(): return Model(True)
def batch(): return torch.ones(2, 4)
''')
    def probe(factory):
        start = time.monotonic()
        result = subprocess.run([cli, "--json", "--no-interactive", "model", "probe", "--model", factory,
                                 "--batch", "fixture:batch"], cwd=project, env=env,
                                capture_output=True, timeout=120)
        return json.loads(result.stdout), len(result.stdout), time.monotonic() - start
    report, _, _ = probe("fixture:short")
    assert report["status"] == "error" and report["diagnostics"][0]["code"] == "FORWARD_FAILED", report
    data = report["data"]
    calls = data["calls"]
    def matches(path):
        return [call for call in calls if path in call["alias_paths"]]
    shared = matches("model.shared")
    assert len(shared) == 2 and [c["invocation_index"] for c in shared] == [1, 2], shared
    assert all("model.alias" in c["alias_paths"] for c in shared), shared
    def prior(path):
        return matches(path)[0]["input_tensor_matches"][0]["prior_outputs"]
    assert shared[0]["invocation_id"] in [item["invocation_id"] for item in prior("model.left")]
    assert shared[1]["invocation_id"] in [item["invocation_id"] for item in prior("model.right")]
    assert prior("model.clone_sink") == [] and prior("model.inplace_sink") == []
    assert not matches("model.unvisited") and data["trace_truncated"] is False
    assert data["tensor_identity_trace"]["scope"] == "partial"
    inspected = subprocess.run([cli, "--json", "--no-interactive", "model", "inspect", "--model", "fixture:short"], cwd=project, env=env, capture_output=True, timeout=90, check=True)
    modules = json.loads(inspected.stdout)["data"]["modules"]
    assert any("model.unvisited" in m.get("paths", []) or m.get("path") == "model.unvisited" for m in modules), modules
    large, report_bytes, seconds = probe("fixture:large")
    assert large["status"] == "error" and large["data"]["trace_truncated"] is True, large
    assert len(large["data"]["calls"]) == 256 and large["data"]["tensor_identity_trace"]["truncated"] is True
    last = large["data"]["failure_location"]["last_observed_module_call"]
    assert "model.head" in last["alias_paths"] and last["invocation_id"] not in {c["invocation_id"] for c in large["data"]["calls"]}
    inputs = "\n1\nfixture:large\n2\n\nfixture:batch\nk\n@2\nb\nq\n"
    tui_start = time.monotonic()
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
                            input=inputs, capture_output=True, text=True, timeout=180, check=True)
    tui_seconds = time.monotonic() - tui_start
    output = result.stdout.replace("\r", "").replace("\n", "")
    assert "The trace was truncated" in output, result.stdout[-7000:]
    assert "Recorded matches may be truncated" in output, result.stdout[-7000:]
    assert "not proof of a producer or dataflow" in output, result.stdout[-7000:]
    print(f"Shared calls, clone, mutation, unvisited module and 40-column truncation: {report_bytes} bytes, {seconds:.3f} s CLI probe; 40-column TUI session {len(result.stdout.encode())} bytes / {tui_seconds:.3f} s")
