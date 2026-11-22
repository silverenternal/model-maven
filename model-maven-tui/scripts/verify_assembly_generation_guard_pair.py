#!/usr/bin/env python3
"""Reject publishing when user factory changes config during codegen verification."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-generate-conflict-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"), COLUMNS="160")
    def run(*args):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                              env=env, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout
        return json.loads(proc.stdout)
    args = ("scaffold", "--input", "x=4", "--output-dim", "2")
    plan = run(*args)
    run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    factory = project / "stateful.py"
    factory.write_text("from torch import nn\ndef encoder(): return nn.Identity()\n")
    run("assembly", "add", "x_encoder", "--factory", "stateful:encoder")
    factory.write_text('''import json
from pathlib import Path
from torch import nn
def encoder():
    counter = Path("counter.txt")
    count = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count + 1))
    if count == 1:
        config = Path(".model-maven/project.json")
        value = json.loads(config.read_text())
        value["concurrent_marker"] = "preserved"
        config.write_text(json.dumps(value))
    return nn.Identity()
''')
    proc = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                          env=env, input="\na\ngenerate\n\ny\nq\n", capture_output=True,
                          text=True, timeout=180, check=True)
    assert "PROJECT_CONFIG_CONFLICT" in proc.stdout, proc.stdout
    assert not (project / "generated/model_assembly.py").exists(), proc.stdout
    assert json.loads((project / ".model-maven/project.json").read_text())["concurrent_marker"] == "preserved"
    assert (project / "counter.txt").read_text() == "2"
    print("TUI reports configuration changed by the factory during generated-code verification; no adapter published")
