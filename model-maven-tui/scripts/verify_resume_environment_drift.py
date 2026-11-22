#!/usr/bin/env python3
"""A real isolated environment changes and is restored between TUI resumes."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-resume-environment-") as directory:
    top = Path(directory)
    project = top / "project"
    project.mkdir()
    venv = top / "python-env"
    subprocess.run(["python3", "-m", "venv", "--system-site-packages", "--without-pip", str(venv)],
                   capture_output=True, text=True, timeout=120, check=True)
    python = str(venv / "bin/python")
    subprocess.run([python, "-c", "import torch"], capture_output=True, text=True, timeout=60, check=True)
    site = Path(subprocess.run([python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                               capture_output=True, text=True, timeout=30, check=True).stdout.strip())
    wrapper = top / "environment-cli"
    wrapper.write_text(f"#!/usr/bin/env python3\nimport os, sys\nos.execv({cli!r}, [{cli!r}, '--python', {python!r}, *sys.argv[1:]])\n")
    wrapper.chmod(0o755)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    def run(*args):
        proc = subprocess.run([str(wrapper), "--json", "--no-interactive", *args], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120, check=True)
        return json.loads(proc.stdout)["data"]
    def session(inputs):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                              env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True).stdout
    def terminal(run_id):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            report = run("runs", "show", run_id)
            if report["record"]["status"] not in ("running", "stopping"):
                return report
            time.sleep(0.25)
        raise AssertionError(f"run did not finish: {run_id}")
    args = ("scaffold", "--input", "x=4", "--output-dim", "2")
    plan = run(*args)
    run(*args, "--apply", "--if-plan", plan["plan_revision"])
    launched = session("\nr\ntrain\n\n\n\n1\n\nn\ny\nq\n")
    match = re.search(r"Run ID:\s*(\S+)", launched)
    assert match, launched
    parent_id = match.group(1)
    parent = terminal(parent_id)
    assert parent["record"]["status"] == "completed", parent
    original_fingerprint = parent["report"]["data"]["runtime"]["environment_fingerprint_sha256"]
    assert len(original_fingerprint) == 64
    fixture = site / "model_maven_drift_fixture-1.0.dist-info"
    fixture.mkdir(parents=True)
    (fixture / "METADATA").write_text("Metadata-Version: 2.1\nName: model-maven-drift-fixture\nVersion: 1.0\n")
    version = subprocess.run([python, "-c", "from importlib.metadata import version; print(version('model-maven-drift-fixture'))"],
                             capture_output=True, text=True, timeout=30, check=True)
    assert version.stdout.strip() == "1.0"
    def resume():
        listing = run("runs", "list")["runs"]
        existing = {item["id"] for item in listing}
        index = next(i for i, item in enumerate(listing, 1) if item["id"] == parent_id)
        output = session(f"\nj\n{index}\nresume\n1\n\ny\nb\nq\n")
        listing = run("runs", "list")["runs"]
        created = [item for item in listing if item["id"] not in existing]
        assert len(created) == 1, (listing, output)
        return created[0]["id"], output
    drift_id, output = resume()
    failed = terminal(drift_id)
    assert failed["record"]["status"] == "failed", failed
    assert "RESUME_RUNTIME_CHANGED" in json.dumps(failed), failed
    changes = failed["report"]["data"]["runtime_changes"]
    assert changes["environment_fingerprint_sha256"]["expected"] == original_fingerprint, changes
    assert changes["environment_fingerprint_sha256"]["actual"] != original_fingerprint, changes
    listing = run("runs", "list")["runs"]
    index = next(i for i, item in enumerate(listing, 1) if item["id"] == drift_id)
    details = session(f"\nj\n{index}\nb\nb\nq\n")
    assert "RESUME_RUNTIME_CHANGED" in details and "Environment inventory SHA-256" in details, details
    assert original_fingerprint in details.replace("\n", "") and changes["environment_fingerprint_sha256"]["actual"] in details.replace("\n", ""), details
    shutil.rmtree(fixture)
    restored_id, output = resume()
    restored = terminal(restored_id)
    assert restored["record"]["status"] == "completed", restored
    assert restored["report"]["data"]["resume_checkpoint"]["sha256"] == parent["checkpoint"]["sha256"], restored
    print("Changed isolated distribution inventory rejects resume and is visible in TUI; restoring inventory allows resume")
