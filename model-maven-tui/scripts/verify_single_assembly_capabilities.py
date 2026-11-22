#!/usr/bin/env python3
"""Distinguish old catalogs from damaged catalogs before single-seam writes."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-single-capability-") as directory:
    project = Path(directory) / "project"
    project.mkdir()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    wrapper = Path(directory) / "catalog-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
if "--describe-interface" in args:
    mode = os.environ["TEST_CATALOG_MODE"]
    if mode == "invalid-json": print("not JSON"); sys.exit(0)
    if mode == "legacy":
        print(json.dumps({{"protocol_version":1,"operation":"cli","status":"error","data":{{}},"diagnostics":[{{"code":"CLI_ARGUMENT_INVALID","message":"unknown option"}}]}})); sys.exit(2)
    result = subprocess.run([{cli!r}] + args, capture_output=True)
    report = json.loads(result.stdout)
    capabilities = report["data"]["capabilities"]
    if mode == "bad-boolean": capabilities["assembly.single_source_evidence"] = "yes"
    if mode == "old":
        capabilities.pop("assembly.single_source_evidence", None)
        capabilities.pop("assembly.single_input_evidence", None)
    print(json.dumps(report)); sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + args)
''')
    wrapper.chmod(0o755)
    def session(inputs, mode=None):
        settings = dict(env)
        if mode: settings["TEST_CATALOG_MODE"] = mode
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper if mode else cli)],
                              cwd=project, env=settings, input=inputs, text=True,
                              capture_output=True, timeout=180, check=True).stdout
    assert "Custom model project created" in session("\nm\nx=4\n2\ny\nq\n")
    (project / "components.py").write_text("from torch import nn\ndef good(): return nn.ReLU()\n")
    config = project / ".model-maven/project.json"
    def bindings(): return json.loads(config.read_text())["model_injections"]
    for mode in ("invalid-json", "bad-boolean"):
        result = session("\na\n\nx_encoder\ncomponents:good\ny\nq\n", mode)
        assert "CLI capability discovery failed" in result and not bindings(), result
    refused = session("\na\n\nx_encoder\ncomponents:good\ninput\nq\n", "old")
    assert "Input evidence unavailable; nothing saved" in refused and not bindings(), refused
    legacy = session("\na\n\nx_encoder\ncomponents:good\ny\nq\n", "legacy")
    assert "source evidence is unavailable" in legacy and "Assembly saved" in legacy, legacy
    assert bindings()["x_encoder"] == "components:good", bindings()
    print("Malformed catalogs block writes; missing and legacy capabilities retain clearly scoped fallback")
