#!/usr/bin/env python3
"""Exercise human-declared training data versions with a local CSV dataset."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-data-versions-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    (project / "train.csv").write_text("x1,x2,y\n1,2,3\n4,5,6\n")
    (project / "data.py").write_text('''import csv
from pathlib import Path
import torch
def batch():
    with Path('train.csv').open(newline='') as stream: rows = list(csv.DictReader(stream))
    return {'args': [], 'kwargs': {'x': torch.tensor([[float(row['x1']), float(row['x2'])] for row in rows])},
            'loss_inputs': {'target': torch.tensor([[float(row['y'])] for row in rows])}}
batch.MODEL_MAVEN_STATELESS = True
''')
    def run(*args, ok=True):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        assert (result.returncode == 0) == ok, report
        return report
    def session(inputs):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True).stdout
    args = ("scaffold", "--input", "x=2", "--output-dim", "1")
    plan = run(*args); run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    args = ("config", "set-batch", "--batch", "data:batch", "--batch-binding", "envelope")
    plan = run(*args); run(*args, "--apply", "--if-current", plan["data"]["from_revision"])
    def trial(version):
        declaration = f"data {version}\n" if version else ""
        output = session("\nr\ntry-wait\n\n\n\n\n\nn\n" + declaration + "y\nq\n")
        match = re.search(r"Run ID:\s*(\S+)", output)
        assert match and "Run started or completed" in output, output
        if version: assert "not an independently verified content hash" in output, output
        return match.group(1)
    left = trial("local-csv/train-v1")
    same = trial("local-csv/train-v1")
    (project / "train.csv").write_text("x1,x2,y\n7,8,9\n2,3,5\n")
    same_claim_changed_data = trial("local-csv/train-v1")
    changed = trial("local-csv/train-v2")
    unknown = trial(None)
    for run_id, version in ((left, "local-csv/train-v1"), (same, "local-csv/train-v1"), (same_claim_changed_data, "local-csv/train-v1"), (changed, "local-csv/train-v2"), (unknown, None)):
        assert run("runs", "show", run_id)["data"]["inputs"]["data_fingerprint"] == version
    def data_comparison(a, b):
        dimensions = run("runs", "compare", a, b)["data"]["dimensions"]
        return next(item for item in dimensions if item["name"] == "data")
    same_report = data_comparison(left, same)
    assert same_report["status"] == "unknown" and same_report["settings_status"] == "same" and same_report["declared_version_status"] == "same", same_report
    changed_content_report = data_comparison(left, same_claim_changed_data)
    assert changed_content_report["status"] == "unknown" and changed_content_report["content_identity"] == "unknown", changed_content_report
    assert data_comparison(left, changed)["status"] == "changed"
    assert data_comparison(unknown, unknown)["status"] == "unknown"
    runs = run("runs", "list")["data"]["runs"]
    left_pos = next(i for i, item in enumerate(runs, 1) if item["id"] == left)
    changed_pos = next(i for i, item in enumerate(runs, 1) if item["id"] == changed)
    same_claim_pos = next(i for i, item in enumerate(runs, 1) if item["id"] == same_claim_changed_data)
    output = session(f"\nj\n{left_pos}\ncompare\n{same_claim_pos}\nb\nb\nq\n")
    assert "data: unknown" in output and "Data settings: same" in output and "Data content identity is not independently verified" in output, output
    wrapper = project / "legacy-comparison-cli"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
if "runs" in args and "compare" in args:
    result = subprocess.run([{cli!r}, *args], capture_output=True, text=True)
    report = json.loads(result.stdout)
    for dimension in report["data"]["dimensions"]:
        if dimension["name"] == "data":
            dimension["status"] = "same"
            for key in ("settings_status", "declared_version_status", "content_identity"): dimension.pop(key, None)
    print(json.dumps(report)); sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}, *args])
''')
    wrapper.chmod(0o755)
    legacy = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project, env=env,
                            input=f"\nj\n{left_pos}\ncompare\n{same_claim_pos}\nb\nb\nq\n", capture_output=True,
                            text=True, timeout=120, check=True).stdout
    assert "data: unknown" in legacy and "Data settings: same" in legacy, legacy
    output = session(f"\nj\n{left_pos}\ncompare\n{changed_pos}\nb\nb\nq\n")
    assert "data: changed" in output and "local-csv/train-v1" in output and "local-csv/train-v2" in output, output
    repeated = run("try", "--repeat", left)
    repeated_id = repeated["data"]["run_id"]
    assert run("runs", "show", repeated_id)["data"]["inputs"]["data_fingerprint"] == "local-csv/train-v1"
    changed_repeat = run("try", "--repeat", left, "--data-fingerprint", "local-csv/train-v2")
    assert "data_fingerprint" in changed_repeat["data"]["repeat_overrides"]
    formal = run("train", "--step", "model_starter:step", "--steps", "1", "--checkpoint-every", "1", "--data-fingerprint", "local-csv/train-v2")
    run_id = formal["data"]["run_id"]
    rejected = run("train", "--resume", run_id, "--additional-steps", "1", "--data-fingerprint", "other", ok=False)
    assert rejected["diagnostics"][0]["code"] == "RESUME_DATA_FINGERPRINT_CHANGED", rejected
    resumed = run("train", "--resume", run_id, "--additional-steps", "1")
    assert run("runs", "show", resumed["data"]["run_id"])["data"]["inputs"]["data_fingerprint"] == "local-csv/train-v2"
    print("Changed CSV with unchanged declared version remains unknown; settings and declarations are shown separately")
    print("Local CSV repeat and resume retain their stated provenance")
