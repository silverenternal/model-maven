#!/usr/bin/env python3
"""Create and probe a dual-input model through actual TUI and CLI binaries."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-custom-pair-") as directory:
    project = Path(directory)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
        env=env, input="\nm\nimage=3,4,4;tabular=5\n3\ny\n2\n\n\nq\n",
        capture_output=True, text=True, timeout=120, check=True)
    assert "Custom model preview" in result.stdout, result.stdout
    assert "Synthetic batch and placeholder loss" in result.stdout, result.stdout
    assert "Custom model project created." in result.stdout, result.stdout
    assert "Observed calls for this batch" in result.stdout, result.stdout
    assert "model.encoders.enc_image" in result.stdout and "model.encoders.enc_tabular" in result.stdout, result.stdout
    source = (project / "model_starter.py").read_text()
    assert "image_encoder" in source and "tabular_encoder" in source and "fusion=None" in source
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["model"] == "model_starter:model" and config["batch"] == "model_starter:batch"
    print("TUI creates a dual-input starter and observes both encoder calls in a real forward")

with tempfile.TemporaryDirectory(prefix="maven-variable-scaffold-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"), COLUMNS="160")
    def session(inputs):
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                                env=env, input=inputs, capture_output=True, text=True, timeout=120, check=True)
        return result.stdout
    output = session("\nm\ntokens:int64=?8;image=3,?4,?5\n3\ny\n2\n\n\nq\n")
    assert "averages that axis" in output and "Custom model project created." in output, output
    assert "Observed calls for this batch" in output, output
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120, check=True)
        return json.loads(result.stdout)
    args = ("config", "set-batch", "--batch", "model_starter:batch", "--batch-param", "batch_size=3",
            "--batch-param", "tokens_dim1=13", "--batch-param", "image_dim2=2", "--batch-param", "image_dim3=7")
    plan = run(*args)
    run(*args, "--apply", "--if-current", plan["data"]["from_revision"])
    output = session("\n2\n\n\nq\n")
    assert "Observed calls for this batch" in output and "FORWARD_FAILED" not in output, output
    print("TUI creates a variable-axis model and re-probes it with a different batch and lengths")

with tempfile.TemporaryDirectory(prefix="maven-custom-module-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"), COLUMNS="160")
    def session(inputs):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True).stdout
    result = session("\nm\na=4;b=4;@module=models.research\n3\ny\n2\n\n\nq\n")
    assert "models/research.py" in result and "Custom model project created." in result and "Observed calls for this batch" in result, result
    source = project / "models/research.py"
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["model"] == "models.research:model" and config["batch"] == "models.research:batch", config
    source.write_text(source.read_text() + "\n# User-owned edit\n")
    edited = source.read_bytes()
    trained = session("\nr\ntry-wait\n\n\n\n\n\n\ny\nq\n")
    assert "models.research:step" in trained and "Run started or completed" in trained, trained
    listing = subprocess.run([cli, "--json", "--no-interactive", "runs", "list"], cwd=project, env=env,
                             capture_output=True, text=True, timeout=30, check=True)
    assert any(run["operation"] == "training_probe" and run["status"] == "completed" for run in json.loads(listing.stdout)["data"]["runs"]), listing.stdout
    route = session("\na\nroute\nb_encoder\na\ny\ny\n2\n\n\nq\n")
    assert "Input route preview" in route and "Observed calls for this batch" in route, route
    assert source.read_bytes() == edited
    print("TUI creates a custom module in a nested directory, probes, trains and edits routes")

with tempfile.TemporaryDirectory(prefix="maven-custom-module-conflict-") as directory:
    project = Path(directory)
    (project / "models").mkdir()
    source = project / "models/research.py"
    source.write_text("# Existing user code\n")
    original = source.read_bytes()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"), COLUMNS="160")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
                            input="\nm\na=4;b=4;@module=models.research\n3\nq\n", capture_output=True, text=True, timeout=120, check=True)
    assert "SCAFFOLD_TARGET_CONFLICT" in result.stdout and "models/research.py" in result.stdout, result.stdout
    assert source.read_bytes() == original and not (project / ".model-maven/project.json").exists()
    for module in ("../escape", "models.class", "a..b"):
        result = subprocess.run([cli, "--json", "--no-interactive", "scaffold", "--input", "a=4", "--module", module],
                                cwd=project, env=env, capture_output=True, text=True, timeout=30)
        report = json.loads(result.stdout)
        assert result.returncode != 0 and report["diagnostics"][0]["code"] == "SCAFFOLD_MODULE_INVALID", report
    assert not (project.parent / "escape.py").exists()
    print("TUI names a conflicting nested source without overwriting; invalid modules are rejected")

with tempfile.TemporaryDirectory(prefix="maven-custom-module-plan-") as directory:
    project = Path(directory)
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", "scaffold", "--input", "a=4", *args], cwd=project,
                                capture_output=True, text=True, timeout=30)
        return result, json.loads(result.stdout)
    result, plan = run("--module", "models.first")
    assert result.returncode == 0, plan
    result, rejected = run("--module", "models.second", "--apply", "--if-plan", plan["data"]["plan_revision"])
    assert result.returncode != 0 and rejected["diagnostics"][0]["code"] == "SCAFFOLD_PLAN_CONFLICT", rejected
    assert not (project / "models").exists() and not (project / ".model-maven/project.json").exists()
    result, batch_plan = run("--batch", "data:first", "--batch-binding", "envelope")
    assert result.returncode == 0 and batch_plan["data"]["plan"]["batch_source"] == "external", batch_plan
    result, rejected = run("--batch", "data:second", "--batch-binding", "envelope", "--apply", "--if-plan", batch_plan["data"]["plan_revision"])
    assert result.returncode != 0 and rejected["diagnostics"][0]["code"] == "SCAFFOLD_PLAN_CONFLICT", rejected
    assert not (project / "model_starter.py").exists()
    result, batch_plan = run("--batch", "data:first", "--batch-param", "size=2")
    assert result.returncode == 0, batch_plan
    result, rejected = run("--batch", "data:first", "--batch-param", "size=3", "--apply", "--if-plan", batch_plan["data"]["plan_revision"])
    assert result.returncode != 0 and rejected["diagnostics"][0]["code"] == "SCAFFOLD_PLAN_CONFLICT", rejected
    result, redacted = run("--batch", "data:first", "--batch-param", 'api_key="not-a-real-key"')
    assert result.returncode == 0 and "not-a-real-key" not in json.dumps(redacted), redacted
    assert redacted["data"]["plan"]["config"]["batch_params"]["api_key"] == "[REDACTED]", redacted
    print("Changing the requested module invalidates the confirmed creation plan")

with tempfile.TemporaryDirectory(prefix="maven-custom-external-batch-") as directory:
    project = Path(directory)
    (project / "data.csv").write_text("a1,a2,a3,a4,b1,b2,target\n1,2,3,4,5,6,0.5\n2,3,4,5,6,7,1.5\n")
    adapter = project / "user_data.py"
    adapter.write_text('''import csv
from pathlib import Path
import torch

def batch(path, scale):
    Path("batch-called").write_text("called")
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {"args": [], "kwargs": {
        "a": torch.tensor([[scale * float(row[f"a{i}"]) for i in range(1, 5)] for row in rows]),
        "b": torch.tensor([[float(row[f"b{i}"]) for i in range(1, 3)] for row in rows])},
        "loss_inputs": {"target": torch.tensor([[float(row["target"])] for row in rows])}}
''')
    original = adapter.read_bytes()
    layers = project / "user_layers.py"
    layers.write_text('''from pathlib import Path
import torch
from torch import nn
def scaled(slope):
    Path("factory-called").write_text("called")
    return nn.LeakyReLU(slope)
def other(): return nn.Identity()
class Fusion(nn.Module):
    def forward(self, values): return torch.cat(values, dim=1)
def fusion(): return Fusion()
def head(): return nn.Linear(6, 1)
''')
    layer_original = layers.read_bytes()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"), COLUMNS="160")
    schema = {"type": "object", "properties": {"slope": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["slope"], "additionalProperties": False}
    registered = subprocess.run([cli, "--json", "--no-interactive", "component", "add", "local/scaled", "--kind", "layer", "--entrypoint", "user_layers:scaled", "--params-schema", json.dumps(schema)], cwd=project, env=env, capture_output=True, text=True, timeout=30)
    assert registered.returncode == 0, registered.stdout
    def scaffold(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", "scaffold", "--input", "a=4", "--input", "b=2", *args], cwd=project, env=env, capture_output=True, text=True, timeout=30)
        return result, json.loads(result.stdout)
    result, seam_plan = scaffold("--inject", "a_encoder=component:local/scaled@0.1.0", "--inject-param", "a_encoder.slope=0.25")
    assert result.returncode == 0, seam_plan
    result, stale = scaffold("--inject", "a_encoder=component:local/scaled@0.1.0", "--inject-param", "a_encoder.slope=0.5", "--apply", "--if-plan", seam_plan["data"]["plan_revision"])
    assert result.returncode != 0 and stale["diagnostics"][0]["code"] == "SCAFFOLD_PLAN_CONFLICT", stale
    result, invalid = scaffold("--inject", "unknown=user_layers:other")
    assert result.returncode != 0 and invalid["diagnostics"][0]["code"] == "SCAFFOLD_SEAM_INVALID", invalid
    assert not (project / "factory-called").exists() and not (project / ".model-maven/project.json").exists()
    def session(inputs):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
                              input=inputs, capture_output=True, text=True, timeout=180, check=True).stdout
    inputs = 'a=4;b=2;@module=models.real;@batch=user_data:batch;@binding=envelope;@batchparam.path="data.csv";@batchparam.scale=2;@seam.a_encoder=component:local/scaled@0.1.0;@seamparam.a_encoder.slope=0.25;@seam.b_encoder=user_layers:other;@seam.fusion=user_layers:fusion;@seam.head=user_layers:head'
    cancelled = session(f"\nm\n{inputs}\n1\nn\nq\n")
    assert "Selected batch factory" in cancelled and "Initial seam bindings" in cancelled and "has not run yet" in cancelled and '"path":"data.csv"' in cancelled.replace(" ", ""), cancelled
    assert not (project / "batch-called").exists() and not (project / "factory-called").exists() and not (project / "models/real.py").exists()
    created = session(f"\nm\n{inputs}\n1\ny\nq\n")
    assert "Custom model project created." in created and not (project / "batch-called").exists() and not (project / "factory-called").exists(), created
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["model"] == "models.real:model" and config["batch"] == "user_data:batch" and config["batch_binding"] == "envelope", config
    assert config["batch_params"] == {"path": "data.csv", "scale": 2}, config
    assert config["model_injections"]["a_encoder"]["component_ref"]["id"] == "local/scaled" and config["model_injections"]["a_encoder"]["params"]["slope"] == 0.25, config
    assert config["model_injections"]["b_encoder"] == "user_layers:other", config
    assert config["model_injections"]["fusion"] == "user_layers:fusion" and config["model_injections"]["head"] == "user_layers:head", config
    assert adapter.read_bytes() == original and layers.read_bytes() == layer_original
    probed = session("\n2\n\n\nq\n")
    assert "Observed calls for this batch" in probed and (project / "batch-called").exists() and (project / "factory-called").exists(), probed
    trial = session("\nr\ntry-wait\n\n\n\n\n\n\ny\nq\n")
    assert "models.real:step" in trial and "user_data:batch" in trial, trial
    runs = subprocess.run([cli, "--json", "--no-interactive", "runs", "list"], cwd=project, env=env,
                          capture_output=True, text=True, timeout=30, check=True)
    assert any(run["operation"] == "training_probe" and run["status"] == "completed" for run in json.loads(runs.stdout)["data"]["runs"]), runs.stdout
    assert adapter.read_bytes() == original
    print("TUI selects an existing CSV-backed batch at creation; preview is non-executing, probe and trial use real fixture data")

with tempfile.TemporaryDirectory(prefix="maven-custom-batch-redaction-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"), COLUMNS="160")
    session = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
        input='\nm\nx=4;@batch=data:batch;@batchparam.api_key="fake-sensitive-value"\n1\nn\nq\n',
        capture_output=True, text=True, timeout=120, check=True)
    assert "[REDACTED]" in session.stdout and "fake-sensitive-value" not in session.stdout, session.stdout
    assert not (project / "model_starter.py").exists()
    print("Batch parameter preview redacts sensitive-looking values while leaving the draft local")
