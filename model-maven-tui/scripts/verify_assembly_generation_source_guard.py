#!/usr/bin/env python3
"""Do not publish an adapter if the generated verification changes project source."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-generation-source-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=120, check=True)
        return json.loads(result.stdout)
    args = ("scaffold", "--input", "x=4", "--output-dim", "2")
    plan = run(*args)
    run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    factory = project / "stateful.py"
    factory.write_text("from torch import nn\ndef encoder(): return nn.Identity()\n")
    run("assembly", "add", "x_encoder", "--factory", "stateful:encoder")
    factory.write_text('''from pathlib import Path
from torch import nn
def encoder():
    counter = Path("counter.txt")
    count = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count + 1))
    if count == 1:
        with Path(__file__).open("a") as stream: stream.write("\\n# changed during verification\\n")
    return nn.Identity()
''')
    output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                            env=env, input="\na\ngenerate\n\ny\nq\n", capture_output=True,
                            text=True, timeout=180, check=True).stdout
    assert "SOURCE_MANIFEST_CONFLICT" in output, output
    assert not (project / "generated/model_assembly.py").exists(), output
    assert "# changed during verification" in factory.read_text() and (project / "counter.txt").read_text() == "2"
    print("Project source changed during generated-code verification: conflict returned, adapter not published")
