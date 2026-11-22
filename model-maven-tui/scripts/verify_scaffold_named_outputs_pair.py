#!/usr/bin/env python3
"""Build and train a two-input, two-output starter through the real binaries."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-named-outputs-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
                            input="\nm\nsignal=4;context=2\nscore=1;classes=3\ny\n2\n\n\nq\n",
                            capture_output=True, text=True, timeout=120, check=True)
    assert "Named outputs" in result.stdout and "reported task losses are unweighted" in result.stdout, result.stdout
    assert "Custom model project created." in result.stdout and "Observed calls for this batch" in result.stdout, result.stdout
    def run(*args, check=True):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        if check: assert proc.returncode == 0 and report["status"] == "success", report
        return proc, report
    _, probe = run("model", "probe")
    outputs = probe["data"]["outputs"]
    assert '"score"' in json.dumps(outputs) and '"classes"' in json.dumps(outputs), outputs
    _, trial = run("try", "--step", "model_starter:step", "--steps", "1")
    summary = json.dumps(trial, ensure_ascii=False)
    assert '"score"' in summary and '"classes"' in summary, trial
    assert "head_score.weight" in summary and "head_classes.weight" in summary, trial
    source = (project / "model_starter.py").read_bytes()
    failed, report = run("scaffold", "--input", "x=4", "--output", "score=1", "--output", "score=2", check=False)
    assert failed.returncode != 0 and report["diagnostics"][0]["code"] == "SCAFFOLD_OUTPUT_INVALID", report
    assert (project / "model_starter.py").read_bytes() == source
    edited = source + b"\n# User-owned custom model change\n"
    (project / "model_starter.py").write_bytes(edited)
    failed, report = run("scaffold", "--input", "x=4", "--output", "score=1", check=False)
    assert failed.returncode != 0 and report["diagnostics"][0]["code"] == "SCAFFOLD_TARGET_CONFLICT", report
    assert (project / "model_starter.py").read_bytes() == edited
    print("TUI builds two named heads, CLI observes both outputs and completes a training step with named losses")

with tempfile.TemporaryDirectory(prefix="maven-named-plan-") as directory:
    project = Path(directory)
    def run(*args):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, capture_output=True, text=True, timeout=120)
        return proc, json.loads(proc.stdout)
    proc, plan = run("scaffold", "--input", "x=4", "--output", "score=1", "--output", "class=3")
    assert proc.returncode == 0, plan
    revision = plan["data"]["plan_revision"]
    proc, rejected = run("scaffold", "--input", "x=4", "--output", "score=1", "--output", "class=4", "--apply", "--if-plan", revision)
    assert proc.returncode != 0 and rejected["diagnostics"][0]["code"] == "SCAFFOLD_PLAN_CONFLICT", rejected
    assert not (project / "model_starter.py").exists()
    print("Output changes invalidate the confirmed plan; existing user code is never overwritten")
