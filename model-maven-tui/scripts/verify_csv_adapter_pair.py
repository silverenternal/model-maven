#!/usr/bin/env python3
"""Create an editable CSV adapter and bind real columns through TUI and CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-csv-adapter-") as directory:
    project = Path(directory)
    csv_file = project / "samples.csv"
    csv_file.write_text("x1,x2,x3,x4,y1,y2\n1,2,3,4,0,1\n4,3,2,1,1,0\n2,2,2,2,0,1\n")
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    def run(*args, ok=True):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env, capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        if ok: assert proc.returncode == 0 and report["status"] == "success", report
        return proc, report
    base = ("scaffold", "--input", "features=4", "--output-dim", "2")
    _, plan = run(*base)
    run(*base, "--apply", "--if-plan", plan["data"]["plan_revision"])
    cancelled = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
        input='\na\nbatch\n:csv\nsamples.csv\n{"features":["x1","x2","x3","x4"]}\n["y1","y2"]\n2\nn\nq\n',
        capture_output=True, text=True, timeout=120, check=True)
    assert "CSV adapter preview" in cancelled.stdout and not (project / "model_csv.py").exists(), cancelled.stdout
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
        input='\na\nbatch\n:csv\nsamples.csv\n{"features":["x1","x2","x3","x4"]}\n["y1","y2"]\n2\ny\ny\nq\n',
        capture_output=True, text=True, timeout=120, check=True)
    assert "CSV adapter preview" in result.stdout and "Batch change preview" in result.stdout, (result.stdout, result.stderr)
    source = (project / "model_csv.py").read_text()
    assert "class CsvBatch" in source
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["batch"] == "model_csv:batch" and config["batch_params"]["path"] == "samples.csv", config
    adapter_check = r'''
import model_csv
args = {"path":"samples.csv", "inputs":{"features":["x1","x2","x3","x4"]}, "targets":["y1","y2"], "batch_size":2}
a = model_csv.CsvBatch()
try: a(**{**args, "inputs":{"features":["missing"]}})
except ValueError: pass
else: raise AssertionError("missing CSV column was accepted")
assert a.state_dict()["cursor"] == 0
first = a(**args)
assert first["kwargs"]["features"].tolist() == [[1,2,3,4],[4,3,2,1]]
saved = a.state_dict()
b = model_csv.CsvBatch(); b.load_state_dict(saved)
assert b(**args)["kwargs"]["features"].tolist() == [[2,2,2,2],[1,2,3,4]]
named = model_csv.CsvBatch()(**{**args, "targets":{"score":["y1"],"class":["y2"]}})
assert named["loss_inputs"]["targets"]["score"].tolist() == [[0],[1]]
assert named["loss_inputs"]["targets"]["class"].tolist() == [[1],[0]]
try: b(**{**args, "batch_size":1})
except ValueError: pass
else: raise AssertionError("changed mapping was accepted")
'''
    subprocess.run([sys.executable, "-c", adapter_check], cwd=project, env=env, capture_output=True, text=True, check=True)
    run("model", "probe")
    run("try", "--step", "model_starter:step", "--steps", "1")
    _, reference = run("--seed", "42", "train", "--step", "model_starter:step", "--steps", "2", "--checkpoint-every", "1")
    _, train = run("--seed", "42", "train", "--step", "model_starter:step", "--steps", "1", "--checkpoint-every", "1")
    run_id = train["data"]["run_id"]
    _, resumed = run("train", "--resume", run_id, "--additional-steps", "1")
    assert resumed["data"]["completed_steps"] == 2, resumed
    assert resumed["data"]["model_state_sha256"] == reference["data"]["model_state_sha256"], (reference, resumed)
    csv_file.write_text(csv_file.read_text().replace("1,2,3,4", "9,2,3,4"))
    proc, rejected = run("train", "--resume", run_id, "--additional-steps", "1", ok=False)
    assert proc.returncode != 0 and "CSV contents or batch mapping changed since the checkpoint" in json.dumps(rejected), rejected
    assert (project / "model_csv.py").read_text() == source
    print("TUI creates and binds numeric CSV adapter; real training resumes and changed data is rejected")

with tempfile.TemporaryDirectory(prefix="maven-csv-template-plan-") as directory:
    project = Path(directory)
    def run(*args):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, capture_output=True, text=True, timeout=30)
        return proc, json.loads(proc.stdout)
    _, plan = run("data", "csv-template")
    revision = plan["data"]["plan_revision"]
    proc, report = run("data", "csv-template", "--apply", "--if-plan", "0" * 64)
    assert proc.returncode != 0 and report["diagnostics"][0]["code"] == "CSV_TEMPLATE_PLAN_CONFLICT" and not (project / "model_csv.py").exists(), report
    proc, report = run("data", "csv-template", "--apply", "--if-plan", revision)
    assert proc.returncode == 0 and report["data"]["committed"], report
    source = (project / "model_csv.py").read_bytes() + b"\n# user changes\n"
    (project / "model_csv.py").write_bytes(source)
    proc, report = run("data", "csv-template")
    assert proc.returncode != 0 and report["diagnostics"][0]["code"] == "CSV_TEMPLATE_EXISTS", report
    assert (project / "model_csv.py").read_bytes() == source
    print("CSV template rejects stale plans and never overwrites user changes")
