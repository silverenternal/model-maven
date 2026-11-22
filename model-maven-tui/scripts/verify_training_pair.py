#!/usr/bin/env python3
"""Run a trial, detached training and stop through the real TUI/CLI pair."""
import json
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-training-pair-") as directory, ExitStack() as cleanup:
    project = Path(directory)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    def command(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=120)
        return result, json.loads(result.stdout)
    def stop_fixture_runs():
        try:
            _, listing = command("runs", "list")
            for run in listing["data"]["runs"]:
                if run["status"] in ("running", "stopping"):
                    command("runs", "stop", run["id"], "--force")
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                _, listing = command("runs", "list")
                if not any(run["status"] in ("running", "stopping") for run in listing["data"]["runs"]):
                    break
                time.sleep(0.2)
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            pass
    cleanup.callback(stop_fixture_runs)
    args = ("scaffold", "--input", "image=3,4,4", "--input", "tabular=5", "--output-dim", "3")
    result, preview = command(*args)
    assert result.returncode == 0, preview
    result, report = command(*args, "--apply", "--if-plan", preview["data"]["plan_revision"])
    assert result.returncode == 0, report
    source = project / "model_starter.py"
    source.write_text(source.read_text().replace("def step(model, batch, context):", "def step(model, batch, context):\n    import time\n    time.sleep(0.15)"))
    inputs = ("\nr\ntry-wait\n\n\n\n\n\n\ny\n"
              "r\ntrain\n\n\n\n50\n\n\ny\n"
              "j\n1\nstop\ny\nb\nb\nq\n")
    output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                            env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True).stdout
    assert output.count("Training plan") == 2 and output.count("Run started or completed") == 2, output
    assert "Run status" in output, output
    _, listing = command("runs", "list")
    runs = listing["data"]["runs"]
    train = next(run for run in runs if run["operation"] == "train")
    run_id = train["id"]
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        _, detail = command("runs", "attach", run_id, "--follow=false")
        status = detail["data"]["record"]["status"]
        if status not in ("running", "stopping"):
            break
        time.sleep(0.3)
    assert status == "interrupted", detail
    assert detail["data"]["record"].get("checkpoint") or list((project / ".model-maven/runs" / run_id / "checkpoints").glob("*")), detail
    _, listing = command("runs", "list")
    index = next(i for i, run in enumerate(listing["data"]["runs"], 1) if run["id"] == run_id)
    _, parent_detail = command("runs", "show", run_id)
    parent_checkpoint = parent_detail["data"]["checkpoint"]
    resume_input = f"\nj\n{index}\nresume\n1\n\ny\nb\nq\n"
    resumed_output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                                    env=env, input=resume_input, capture_output=True, text=True, timeout=180, check=True).stdout
    assert "Resume checkpoint" in resumed_output and "SHA-256:" in resumed_output, resumed_output
    assert "Recorded Python / PyTorch" in resumed_output and "not a current comparison" in resumed_output, resumed_output
    _, listing = command("runs", "list")
    resumed = next((run for run in listing["data"]["runs"] if run["id"] != run_id and run["operation"] == "train"), None)
    assert resumed is not None, (listing, resumed_output)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        _, resumed_detail = command("runs", "attach", resumed["id"], "--follow=false")
        resumed_status = resumed_detail["data"]["record"]["status"]
        if resumed_status not in ("running", "stopping"):
            break
        time.sleep(0.3)
    assert resumed_status == "completed", resumed_detail
    _, saved_resume = command("runs", "show", resumed["id"])
    provenance = saved_resume["data"]["report"]["data"]["resume_checkpoint"]
    assert provenance["sha256"] == parent_checkpoint["sha256"] and provenance["completed_steps"] == parent_checkpoint["completed_steps"], provenance
    assert provenance["confirmed_sha256"] == parent_checkpoint["sha256"] and provenance["loaded_from_verified_copy"] is True, provenance
    _, listing = command("runs", "list")
    resumed_index = next(i for i, run in enumerate(listing["data"]["runs"], 1) if run["id"] == resumed["id"])
    reopened = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input=f"\nj\n{resumed_index}\nb\nb\nq\n", capture_output=True,
                              text=True, timeout=90, check=True)
    assert "Resumed checkpoint" in reopened.stdout and parent_checkpoint["sha256"] in reopened.stdout.replace("\n", ""), reopened.stdout
    print("TUI runs a trial, launches background training, reconnects, stops at a checkpoint and resumes")
    source_before = source.read_bytes()
    with source.open("a") as stream:
        stream.write("\n# changed after the original training run\n")
    _, listing = command("runs", "list")
    existing = {run["id"] for run in listing["data"]["runs"]}
    index = next(i for i, run in enumerate(listing["data"]["runs"], 1) if run["id"] == run_id)
    drifted = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                             env=env, input=f"\nj\n{index}\nresume\n1\n\ny\nb\nq\n",
                             capture_output=True, text=True, timeout=180, check=True)
    _, listing = command("runs", "list")
    created = [run for run in listing["data"]["runs"] if run["id"] not in existing]
    assert len(created) == 1, (listing, drifted.stdout)
    drift_id = created[0]["id"]
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        _, drift_detail = command("runs", "show", drift_id)
        if drift_detail["data"]["record"]["status"] not in ("running", "stopping"):
            break
        time.sleep(0.3)
    assert drift_detail["data"]["record"]["status"] == "failed" and "RESUME_SOURCE_CHANGED" in json.dumps(drift_detail), drift_detail
    _, listing = command("runs", "list")
    drift_index = next(i for i, run in enumerate(listing["data"]["runs"], 1) if run["id"] == drift_id)
    visible = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                             env=env, input=f"\nj\n{drift_index}\nb\nb\nq\n",
                             capture_output=True, text=True, timeout=90, check=True)
    assert "RESUME_SOURCE_CHANGED" in visible.stdout, visible.stdout
    source.write_bytes(source_before)
    print("TUI resume rejects source drift and shows the precise saved failure reason after reconnecting")
    wrapper_dir = Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix="maven-resume-race-")))
    race_wrapper = wrapper_dir / "race-cli.py"
    race_wrapper.write_text(f'''#!/usr/bin/env python3
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "--resume" in args and "--if-checkpoint-sha256" in args:
    run_id = args[args.index("--resume") + 1]
    directory = Path(".model-maven/runs") / run_id / "checkpoints"
    path = directory / "latest.json"
    manifest = json.loads(path.read_text())
    checkpoint = directory / manifest["file"]
    with checkpoint.open("ab") as stream: stream.write(b"race-test")
    manifest["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
os.execv({cli!r}, [{cli!r}] + args)
''')
    race_wrapper.chmod(0o755)
    _, listing = command("runs", "list")
    before_race = {run["id"] for run in listing["data"]["runs"]}
    index = next(i for i, run in enumerate(listing["data"]["runs"], 1) if run["id"] == run_id)
    raced = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(race_wrapper)], cwd=project,
                           env=env, input=f"\nj\n{index}\nresume\n1\n\ny\nb\nq\n",
                           capture_output=True, text=True, timeout=180, check=True)
    assert "Resume checkpoint" in raced.stdout, raced.stdout
    _, listing = command("runs", "list")
    race_runs = [run for run in listing["data"]["runs"] if run["id"] not in before_race]
    assert len(race_runs) == 1, (listing, raced.stdout)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        _, race_detail = command("runs", "show", race_runs[0]["id"])
        if race_detail["data"]["record"]["status"] not in ("running", "stopping"):
            break
        time.sleep(0.3)
    assert race_detail["data"]["record"]["status"] == "failed", race_detail
    assert "CHECKPOINT_CHANGED" in json.dumps(race_detail), race_detail
    print("A checkpoint replaced after TUI confirmation is rejected before loading its state")
    _, listing = command("runs", "list")
    existing_ids = {run["id"] for run in listing["data"]["runs"]}
    wrapper = project / "lost-receipt.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, subprocess, sys
if "train" in sys.argv and "--detach" in sys.argv and "--resume" not in sys.argv:
    result = subprocess.run([{cli!r}] + sys.argv[1:], capture_output=True)
    if result.returncode == 0:
        print("invalid receipt")
        sys.exit(0)
    sys.stdout.buffer.write(result.stdout)
    sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                            env=env, input="\nr\ntrain\n\n\n\n50\n\n\ny\nq\n",
                            capture_output=True, text=True, timeout=180, check=True)
    assert "Checking saved runs; no automatic retry." in result.stdout, result.stdout
    _, listing = command("runs", "list")
    new_runs = [run for run in listing["data"]["runs"] if run["id"] not in existing_ids]
    assert len(new_runs) == 1 and new_runs[0]["id"] in result.stdout, (listing, result.stdout)
    print("Lost launch receipt: TUI queries actual runs and does not start a duplicate")
