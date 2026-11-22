#!/usr/bin/env python3
"""Compare, evaluate and export real training runs through the TUI."""
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-artifacts-pair-") as directory:
    project = Path(directory) / "project"
    project.mkdir()
    destination = Path(directory) / "exported"
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        assert result.returncode == 0, report
        return report
    args = ("scaffold", "--input", "image=3,4,4", "--input", "tabular=5", "--output-dim", "3")
    preview = run(*args)
    run(*args, "--apply", "--if-plan", preview["data"]["plan_revision"])
    (project / "eval_fixture.py").write_text('''import torch
from pathlib import Path
def data(size=0):
    assert size == 10
    return {"image": torch.ones(size, 3, 4, 4), "tabular": torch.ones(size, 5)}
def evaluate(model, data, context, scale=0):
    Path("evaluator-ran").write_text("yes")
    assert scale == 3 and context.max_samples == 10
    output = model(**data) * scale
    context.report_samples(10)
    for index in range(10):
        context.report_prediction(f"synthetic-{index}", output[index].tolist(), target=[index] * 3)
    return {"metrics": {"mean_abs": float(output.abs().mean())}}
''')
    first = run("train", "--step", "model_starter:step", "--steps", "1", "--checkpoint-every", "1")
    second = run("train", "--step", "model_starter:step", "--steps", "1", "--checkpoint-every", "1")
    ids = {first["data"]["run_id"], second["data"]["run_id"]}
    assert len(ids) == 2
    listing = run("runs", "list")["data"]["runs"]
    selected = next(i for i, item in enumerate(listing, 1) if item["id"] == second["data"]["run_id"])
    other = next(i for i, item in enumerate(listing, 1) if item["id"] == first["data"]["run_id"])
    inputs = (f"\nj\n{selected}\ncompare\n{other}\n"
              "evaluate\neval_fixture:evaluate\neval_fixture:data\ntest\nfixture-v1\n10\n10\n{\"scale\":3}\n{\"size\":10}\ny\n"
              f"export\n{destination}\ny\nexport\n{destination}\n\nb\nb\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                            env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    output = result.stdout
    assert "Run comparison" in output and "samples: unknown" in output and "data: unknown" in output, output
    assert "Evaluation result" in output and "mean_abs" in output and "synthetic-0" in output, output
    assert "Export completed" in output, output
    evaluation = next(item for item in run("runs", "list")["data"]["runs"] if item["operation"] == "evaluate")
    saved = run("runs", "show", evaluation["id"])["data"]["report"]["data"]
    assert saved["source_run_id"] == second["data"]["run_id"], saved
    parent_checkpoint = run("runs", "show", second["data"]["run_id"])["data"]["checkpoint"]
    assert saved["checkpoint"]["sha256"] == parent_checkpoint["sha256"]
    assert "Checkpoint file:" in output and parent_checkpoint["sha256"] in output.replace("\n", ""), output
    assert saved["evaluation"]["sample_previews"]["reported"] == 10, saved
    assert saved["provenance"]["data_identity"]["data_fingerprint"] == "fixture-v1", saved
    (project / "evaluator-ran").unlink()
    receipt_wrapper = Path(directory) / "alter-evaluation-receipt.py"
    receipt_wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
if "evaluate" in args:
    result = subprocess.run([{cli!r}, *args], capture_output=True)
    report = json.loads(result.stdout)
    if report["status"] == "success":
        identity = report["data"]["provenance"]["data_identity"]
        mode = os.environ["EVAL_RECEIPT_MODE"]
        if mode == "evaluator": report["data"]["evaluation"]["entrypoint"] = "other:evaluator"
        if mode == "data": identity["entrypoint"] = "other:data"
        if mode == "version": identity["data_fingerprint"] = "other-version"
        if mode == "limit": report["data"]["evaluation"]["sample_limit"] = 2
        if mode == "missing": identity.pop("parameters")
    print(json.dumps(report)); sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + args)
''')
    receipt_wrapper.chmod(0o755)
    for mode in ("evaluator", "data", "version", "limit", "missing"):
        listing = run("runs", "list")["data"]["runs"]
        selected = next(i for i, item in enumerate(listing, 1) if item["id"] == second["data"]["run_id"])
        evaluation_input = f'\nj\n{selected}\nevaluate\neval_fixture:evaluate\neval_fixture:data\ntest\nfixture-v1\n10\n10\n{{"scale":3}}\n{{"size":10}}\ny\nb\nb\nq\n'
        altered = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(receipt_wrapper)], cwd=project,
                                 env=dict(env, EVAL_RECEIPT_MODE=mode), input=evaluation_input,
                                 capture_output=True, text=True, timeout=120, check=True).stdout
        assert "Evaluation receipt unverified" in altered and "Evaluation result" not in altered, (mode, altered)
        assert "Inspect the saved run before retrying" in altered and (project / "evaluator-ran").exists(), (mode, altered)
        (project / "evaluator-ran").unlink()
    print("Mismatched or missing evaluator and data identity are not attributed to the requested evaluation")
    checkpoint_dir = project / ".model-maven/runs" / second["data"]["run_id"] / "checkpoints"
    latest_path = checkpoint_dir / "latest.json"
    checkpoint_path = checkpoint_dir / parent_checkpoint["file"]
    original_manifest, original_checkpoint = latest_path.read_bytes(), checkpoint_path.read_bytes()
    wrapper = Path(directory) / "eval-race.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "evaluate" in args and "--if-checkpoint-sha256" in args:
    path = Path({str(latest_path)!r})
    manifest = json.loads(path.read_text())
    checkpoint = path.parent / manifest["file"]
    with checkpoint.open("ab") as stream: stream.write(b"race-test")
    manifest["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
os.execv({cli!r}, [{cli!r}] + args)
''')
    wrapper.chmod(0o755)
    listing = run("runs", "list")["data"]["runs"]
    selected = next(i for i, item in enumerate(listing, 1) if item["id"] == second["data"]["run_id"])
    race_inputs = f'\nj\n{selected}\nevaluate\neval_fixture:evaluate\neval_fixture:data\ntest\nfixture-v1\n10\n10\n{{"scale":3}}\n{{"size":10}}\ny\nb\nb\nq\n'
    try:
        raced = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                               env=env, input=race_inputs, capture_output=True, text=True, timeout=120, check=True)
        assert "CHECKPOINT_CHANGED" in raced.stdout and not (project / "evaluator-ran").exists(), raced.stdout
    finally:
        checkpoint_path.write_bytes(original_checkpoint)
        latest_path.write_bytes(original_manifest)
    print("Evaluation rejects a checkpoint replaced after confirmation before running the evaluator")
    export_wrapper = Path(directory) / "export-race.py"
    export_wrapper.write_text(f'''#!/usr/bin/env python3
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "export" in args and "--if-checkpoint-sha256" in args:
    path = Path({str(latest_path)!r})
    manifest = json.loads(path.read_text())
    checkpoint = path.parent / manifest["file"]
    with checkpoint.open("ab") as stream: stream.write(b"export-race-test")
    manifest["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
os.execv({cli!r}, [{cli!r}] + args)
''')
    export_wrapper.chmod(0o755)
    listing = run("runs", "list")["data"]["runs"]
    selected = next(i for i, item in enumerate(listing, 1) if item["id"] == second["data"]["run_id"])
    race_destination = Path(directory) / "race-export"
    try:
        raced = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(export_wrapper)], cwd=project,
                               env=env, input=f"\nj\n{selected}\nexport\n{race_destination}\ny\nb\nb\nq\n",
                               capture_output=True, text=True, timeout=120, check=True)
        assert "CHECKPOINT_CHANGED" in raced.stdout and not race_destination.exists(), raced.stdout
    finally:
        checkpoint_path.write_bytes(original_checkpoint)
        latest_path.write_bytes(original_manifest)
    print("Export rejects a checkpoint replaced after confirmation without publishing a destination")
    listing = run("runs", "list")["data"]["runs"]
    eval_index = next(i for i, item in enumerate(listing, 1) if item["id"] == evaluation["id"])
    browsed = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                             env=env, input=f"\nj\n{eval_index}\nnext\n9\nb\nb\nq\n", capture_output=True,
                             text=True, timeout=90, check=True)
    assert "Saved evaluation predictions" in browsed.stdout and "Prediction detail" in browsed.stdout, browsed.stdout
    assert "synthetic-8" in browsed.stdout and '"target":[8,8,8]' in browsed.stdout, browsed.stdout
    assert second["data"]["run_id"] in browsed.stdout and "fixture-v1" in browsed.stdout and "Checkpoint:" in browsed.stdout, browsed.stdout
    print("Saved evaluation reopens with linked metrics and individually browsable predictions")
    assert (destination / "manifest.json").is_file()
    assert json.loads((destination / "manifest.json").read_text())["checkpoint"]["sha256"] == parent_checkpoint["sha256"]
    readme = (destination / "README.md").read_text()
    assert "checkpoint_directory" in readme and "from_checkpoint" in readme, readme
    # Compare a real original-project continuation with the isolated exported helper.
    continued = run("train", "--resume", second["data"]["run_id"], "--additional-steps", "1")
    expected_hash = continued["data"]["model_state_sha256"]
    continued_twice = run("train", "--resume", second["data"]["run_id"], "--additional-steps", "2")
    expected_two_hash = continued_twice["data"]["model_state_sha256"]
    resume_env = env.copy()
    resume_env.pop("PYTHONPATH", None)
    resume_env.pop("PYTHONHOME", None)
    hidden_project = project.with_name("temporarily-hidden-original")
    project.rename(hidden_project)
    try:
        isolated = subprocess.run([sys.executable, "-c", "import restore_model as r; out=r.resume_managed_steps(1); assert out['completed_steps']==2; assert r.worker.model_state_digest(r.model)==__import__('sys').argv[1];\ntry: r.resume_managed_steps(1)\nexcept RuntimeError: pass\nelse: raise AssertionError('repeat unexpectedly accepted')\nprint('CONTINUATION_MATCH')", expected_hash],
                                  cwd=destination, env=resume_env, capture_output=True, text=True, timeout=120)
        first_dir = destination.parent / "continued-one"
        second_dir = destination.parent / "continued-two"
        first_code = "import restore_model as r,sys; x=r.resume_managed_steps(1,checkpoint_directory=sys.argv[1]); assert x['continuation_checkpoint']['completed_steps']==2; print('SAVED_ONE')"
        saved_one = subprocess.run([sys.executable, "-c", first_code, str(first_dir)], cwd=destination, env=resume_env,
                                   capture_output=True, text=True, timeout=120)
        assert saved_one.returncode == 0 and "SAVED_ONE" in saved_one.stdout, (saved_one.stdout, saved_one.stderr)
        second_code = "import restore_model as r,sys; x=r.resume_managed_steps(1,from_checkpoint=sys.argv[1],checkpoint_directory=sys.argv[2]); assert x['continuation_checkpoint']['completed_steps']==3; assert r.worker.model_state_digest(r.model)==sys.argv[3]; print('SAVED_TWO_MATCH')"
        saved_two = subprocess.run([sys.executable, "-c", second_code, str(first_dir), str(second_dir), expected_two_hash],
                                   cwd=destination, env=resume_env, capture_output=True, text=True, timeout=120)
        assert saved_two.returncode == 0 and "SAVED_TWO_MATCH" in saved_two.stdout, (saved_two.stdout, saved_two.stderr)
        existing = subprocess.run([sys.executable, "-c", first_code, str(first_dir)], cwd=destination, env=resume_env,
                                  capture_output=True, text=True, timeout=120)
        assert existing.returncode != 0 and "FileExistsError" in existing.stderr
        damaged_dir = destination.parent / "damaged"
        shutil.copytree(first_dir, damaged_dir)
        latest = json.loads((damaged_dir / "latest.json").read_text())
        with (damaged_dir / latest["file"]).open("ab") as stream: stream.write(b"corrupt")
        rejected = subprocess.run([sys.executable, "-c", second_code, str(damaged_dir), str(destination.parent / "must-not-exist"), expected_two_hash],
                                  cwd=destination, env=resume_env, capture_output=True, text=True, timeout=120)
        assert rejected.returncode != 0 and not (destination.parent / "must-not-exist").exists(), (rejected.stdout, rejected.stderr)
    finally:
        hidden_project.rename(project)
    assert isolated.returncode == 0 and "CONTINUATION_MATCH" in isolated.stdout, (isolated.stdout, isolated.stderr)
    print("Exported managed CPU continuation matches original checkpoint continuation weights")
    print("Two independent saved continuations match the original run; existing outputs and corrupted checkpoints are rejected")
    original_manifest = (destination / "manifest.json").read_bytes()
    (destination / "keep.txt").write_text("existing user content")
    listing = run("runs", "list")["data"]["runs"]
    selected = next(i for i, item in enumerate(listing, 1) if item["id"] == second["data"]["run_id"])
    conflict_input = f"\nj\n{selected}\nexport\n{destination}\ny\nb\nb\nq\n"
    conflict = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input=conflict_input, capture_output=True, text=True, timeout=120, check=True)
    assert "EXPORT_FAILED" in conflict.stdout, conflict.stdout
    assert (destination / "manifest.json").read_bytes() == original_manifest
    assert (destination / "keep.txt").read_text() == "existing user content"
    restore_env = env.copy()
    restore_env.pop("PYTHONPATH", None)
    restored = subprocess.run([sys.executable, "restore_model.py"], cwd=destination, env=restore_env,
                              capture_output=True, text=True, timeout=120)
    assert restored.returncode == 0, (restored.stdout, restored.stderr)
    print("TUI compares two runs, evaluates predictions and exports a checkpoint restored independently")
