#!/usr/bin/env python3
"""Verify missing or malformed CLI capabilities stop training before launch."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
with tempfile.TemporaryDirectory(prefix="maven-training-capabilities-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), REAL_MAVEN_CLI=cli, COLUMNS="160")
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, text=True, capture_output=True, timeout=120, check=True)
        return json.loads(result.stdout)
    args = ("scaffold", "--input", "x=4", "--output-dim", "2")
    preview = run(*args)
    run(*args, "--apply", "--if-plan", preview["data"]["plan_revision"])
    run("layer", "set", "model.head", "--param", "bias=false", "--require-forward")
    wrapper = project / "capability-wrapper.py"
    wrapper.write_text('''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with Path("calls.jsonl").open("a") as stream: stream.write(json.dumps(args) + "\\n")
if "--describe-interface" in args:
    result = subprocess.run([os.environ["REAL_MAVEN_CLI"], *args], capture_output=True)
    report = json.loads(result.stdout)
    mode = os.environ["CAP_MODE"]
    if mode == "bad-late":
        marker = Path("catalog-count")
        count = int(marker.read_text()) if marker.exists() else 0
        marker.write_text(str(count + 1))
        if count >= 1: print("not JSON"); sys.exit(0)
    elif mode == "bad-config":
        report["data"]["capabilities"]["training.conditional_config_start"] = "yes"
    else:
        report["data"]["capabilities"].pop(mode, None)
    print(json.dumps(report)); sys.exit(result.returncode)
os.execv(os.environ["REAL_MAVEN_CLI"], [os.environ["REAL_MAVEN_CLI"], *args])
''')
    wrapper.chmod(0o755)
    def session(mode, inputs):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                              env=dict(env, CAP_MODE=mode), input=inputs, capture_output=True,
                              text=True, timeout=120, check=True).stdout
    for mode, inputs, expected in [
        ("training.conditional_config_start", "\nr\ntrain\n\n\n\n1\n\nn\nq\n", "training.conditional_config_start"),
        ("bad-config", "\nr\ntrain\n\n\n\n1\n\nn\nq\n", "Could not verify CLI capabilities"),
        ("training.conditional_supervision", "\nr\ntrain\n\n\n\n1\n\ny\nq\n", "training.conditional_supervision"),
        ("training.conditional_variant_start", "\ns\n4\n1\nb\nr\ntrain\n\n\n\n1\n\nn\nq\n", "training.conditional_variant_start"),
        ("training.data_fingerprint", "\nr\ntrain\n\n\n\n1\n\nn\ndata dataset-v1\ny\nq\n", "training.data_fingerprint"),
        ("bad-late", "\nr\ntrain\n\n\n\n1\n\nn\ny\nq\n", "Could not verify CLI capabilities"),
    ]:
        output = session(mode, inputs)
        assert expected in output, (mode, output)
        assert not any(item["operation"] == "train" for item in run("runs", "list")["data"]["runs"]), output
        pending = project / ".model-maven/tui-pending-starts"
        assert not pending.exists() or not list(pending.iterdir()), (mode, output)
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("train" in args and "--detach" in args for args in calls), calls
    output = session("training.data_fingerprint", "\nr\ntry-wait\n\n\n\n\n\nn\ndata dataset-v1\ny\nclear\ny\nq\n")
    assert "training.data_fingerprint" in output and "Run started or completed" in output, output
    run_id = re.search(r"Run ID:\s*(\S+)", output)
    assert run_id, output
    assert run("runs", "show", run_id.group(1))["data"]["inputs"]["data_fingerprint"] is None
    print("Missing config, supervision, variant or data-version capabilities and malformed catalogs refuse training before launch")
