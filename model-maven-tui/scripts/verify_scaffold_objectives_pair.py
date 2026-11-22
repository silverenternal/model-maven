#!/usr/bin/env python3
"""Create a mixed regression/classification model and train on CSV labels."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-objectives-") as directory:
    project = Path(directory)
    data = project / "train.csv"
    data.write_text("x1,x2,x3,score,label\n1,2,3,0.5,0\n4,5,6,1.5,2\n")
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="180")
    def session(inputs):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
                              input=inputs, capture_output=True, text=True, timeout=120, check=True).stdout
    def run(*args, ok=True):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        if ok: assert proc.returncode == 0 and report["status"] == "success", report
        return proc, report
    output = session("\nm\na=2;b=1\nscore=1:mse@0.25;classes=3:ce@2\ny\nq\n")
    assert "Named outputs" in output and "cross entropy uses integer class IDs" in output and "Custom model project created." in output, output
    source = (project / "model_starter.py").read_text()
    assert 'nn.functional.cross_entropy(prediction["classes"], targets["classes"])' in source
    checked = subprocess.run(["python3", "-c", '''import torch, model_starter as s
m = s.model(); result = s.step(m, s.batch(), None)
expected = 0.25 * result["losses"]["score"] + 2 * result["losses"]["classes"]
assert torch.allclose(result["loss"], expected)
result["loss"].backward()
assert m.head.heads["head_score"].weight.grad is not None
assert m.head.heads["head_classes"].weight.grad is not None
'''], cwd=project, env=env, capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, (checked.stdout, checked.stderr)
    output = session('\na\nbatch\n:csv\ntrain.csv\n{"a":["x1","x2"],"b":["x3"]}\n{"score":["score"],"classes":{"columns":["label"],"shape":[],"dtype":"int64"}}\n2\ny\ny\nq\n')
    assert "Batch change preview" in output, output
    _, trial = run("try", "--step", "model_starter:step", "--steps", "1")
    text = json.dumps(trial)
    assert '"score"' in text and '"classes"' in text and "head_score.weight" in text and "head_classes.weight" in text, trial
    data.write_text(data.read_text().replace("1.5,2", "1.5,7"))
    proc, invalid = run("try", "--step", "model_starter:step", "--steps", "1", ok=False)
    assert proc.returncode != 0 and invalid["data"]["completed_steps"] == 0, invalid
    assert "out of bounds" in json.dumps(invalid).lower(), invalid
    print("Mixed regression/classification model trains on CSV targets and rejects an out-of-range class")

with tempfile.TemporaryDirectory(prefix="maven-single-ce-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="180")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
        input="\nm\nx=4\nce:3\ny\nq\n", capture_output=True, text=True, timeout=120, check=True)
    assert "Cross entropy expects logits" in result.stdout and "Custom model project created." in result.stdout, result.stdout
    proc = subprocess.run([cli, "--json", "--no-interactive", "try", "--step", "model_starter:step", "--steps", "1"], cwd=project, env=env, capture_output=True, text=True, timeout=120)
    report = json.loads(proc.stdout)
    assert proc.returncode == 0 and report["status"] == "success", report
    before = (project / "model_starter.py").read_bytes()
    proc = subprocess.run([cli, "--json", "--no-interactive", "scaffold", "--input", "x=4", "--output", "bad=1:ce"], cwd=project, capture_output=True, text=True, timeout=30)
    invalid = json.loads(proc.stdout)
    assert proc.returncode != 0 and invalid["diagnostics"][0]["code"] == "SCAFFOLD_OUTPUT_INVALID", invalid
    assert (project / "model_starter.py").read_bytes() == before
    print("Single classification scaffold trains with generated integer labels")

with tempfile.TemporaryDirectory(prefix="maven-objective-plan-") as directory:
    project = Path(directory)
    def plan(*args):
        proc = subprocess.run([cli, "--json", "--no-interactive", "scaffold", "--input", "x=4", *args], cwd=project, capture_output=True, text=True, timeout=30)
        return proc, json.loads(proc.stdout)
    _, mse = plan("--output", "class=3:mse")
    revision = mse["data"]["plan_revision"]
    proc, rejected = plan("--output", "class=3:ce", "--apply", "--if-plan", revision)
    assert proc.returncode != 0 and rejected["diagnostics"][0]["code"] == "SCAFFOLD_PLAN_CONFLICT", rejected
    assert not (project / "model_starter.py").exists()
    _, weighted = plan("--output", "class=3:ce@0.5")
    assert weighted["data"]["plan"]["output"]["heads"][0]["weight"] == 0.5, weighted
    proc, rejected = plan("--output", "class=3:ce@2", "--apply", "--if-plan", weighted["data"]["plan_revision"])
    assert proc.returncode != 0 and rejected["diagnostics"][0]["code"] == "SCAFFOLD_PLAN_CONFLICT", rejected
    for weight in ("0", "-1", "NaN", "inf"):
        proc, rejected = plan("--output", f"class=3:ce@{weight}")
        assert proc.returncode != 0 and rejected["diagnostics"][0]["code"] == "SCAFFOLD_OUTPUT_INVALID", rejected
    print("Changing an objective invalidates the confirmed scaffold plan")
