#!/usr/bin/env python3
"""Bind shaped, typed CSV fields to a real starter and validate training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-csv-typed-") as directory:
    project = Path(directory)
    (project / "samples.csv").write_text("p0,p1,p2,p3,t0,t1,valid,y0,y1\n1,2,3,4,9007199254740993,2,true,0,1\n4,3,2,1,5,6,0,1,0\n")
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="180")
    def run(*args):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env, capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        assert proc.returncode == 0 and report["status"] == "success", report
        return report
    base = ("scaffold", "--input", "image=2,2", "--input", "tokens:int64=2", "--input", "mask:bool=1", "--output-dim", "2")
    plan = run(*base)
    run(*base, "--apply", "--if-plan", plan["data"]["plan_revision"])
    mapping = {"image":{"columns":["p0","p1","p2","p3"],"shape":[2,2]}, "tokens":{"columns":["t0","t1"],"dtype":"int64"}, "mask":{"columns":["valid"],"dtype":"bool"}}
    inputs = "\na\nbatch\n:csv\nsamples.csv\n" + json.dumps(mapping) + '\n["y0","y1"]\n2\ny\ny\nq\n'
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env, input=inputs, capture_output=True, text=True, timeout=120, check=True)
    assert "CSV adapter preview" in result.stdout and "Batch change preview" in result.stdout, (result.stdout, result.stderr)
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["batch_params"]["inputs"] == mapping, config
    check = r'''
import model_csv, torch
inputs = {"image":{"columns":["p0","p1","p2","p3"],"shape":[2,2]}, "tokens":{"columns":["t0","t1"],"dtype":"int64"}, "mask":{"columns":["valid"],"dtype":"bool"}}
b = model_csv.CsvBatch()
v = b(path="samples.csv", inputs=inputs, targets=["y0","y1"], batch_size=2)
assert v["kwargs"]["image"].shape == (2,2,2)
assert v["kwargs"]["tokens"].dtype == torch.int64 and v["kwargs"]["tokens"][0,0].item() == 9007199254740993
assert v["kwargs"]["mask"].dtype == torch.bool and v["kwargs"]["mask"].tolist() == [[True],[False]]
t = model_csv.CsvBatch()(path="samples.csv", inputs=inputs, targets={"$tensor":{"columns":["y0"],"shape":[],"dtype":"int64"}}, batch_size=2)
assert t["loss_inputs"]["target"].dtype == torch.int64 and t["loss_inputs"]["target"].shape == (2,)
extra = model_csv.CsvBatch()(path="samples.csv", inputs={"double":{"columns":["p0"],"dtype":"float64"}, "half":{"columns":["p1"],"dtype":"float16"}}, targets={"score":{"columns":["y0"],"shape":[],"dtype":"int64"}}, batch_size=2)
assert extra["kwargs"]["double"].dtype == torch.float64 and extra["kwargs"]["half"].dtype == torch.float16
assert extra["loss_inputs"]["targets"]["score"].dtype == torch.int64 and extra["loss_inputs"]["targets"]["score"].shape == (2,)
for bad in [{"image":{"columns":["p0","p1"],"shape":[3]}}, {"image":{"columns":["p0"],"dtype":"complex64"}}, {"image":{"columns":["p0"],"dtype":"bool"}}]:
    fresh = model_csv.CsvBatch()
    try: fresh(path="samples.csv", inputs=bad, targets=["y0"], batch_size=2)
    except ValueError: assert fresh.state_dict()["cursor"] == 0
    else: raise AssertionError(f"invalid typed mapping accepted: {bad}")
'''
    subprocess.run([sys.executable, "-c", check], cwd=project, env=env, capture_output=True, text=True, check=True)
    run("model", "probe")
    # The default identity encoder feeds raw numbers to a float32 head; use
    # bounded values for the optimizer test, while the separate check above
    # proves that CSV parsing preserves large integers exactly.
    data_path = project / "samples.csv"
    data_path.write_text(data_path.read_text().replace("9007199254740993", "7"))
    run("try", "--step", "model_starter:step", "--steps", "1")
    print("TUI binds shaped float, exact int64 and bool CSV fields; probe and training succeed")
