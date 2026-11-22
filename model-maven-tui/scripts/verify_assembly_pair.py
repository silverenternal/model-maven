#!/usr/bin/env python3
"""Create a dual-input model and edit its assembly through the real TUI."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-assembly-pair-") as directory:
    project = Path(directory)
    (project / "components.py").write_text("from torch import nn\ndef good(): return nn.ReLU()\ndef bad(): return nn.Linear(2, 2)\n")
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    inputs = ("\nm\nimage=3,4,4;tabular=5\n3\ny\n"
              "a\n\nimage_encoder\ncomponents:bad\n"
              "a\n\nimage_encoder\ncomponents:good\n\n"
              "a\n\nimage_encoder\ncomponents:good\ny\n"
              "2\n\n\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                            env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    output = result.stdout
    assert "FORWARD_FAILED" in output, output
    assert output.count("Assembly preview") == 2, output
    assert "Assembly saved; previous diagnosis has been cleared." in output, output
    assert "model.encoders.enc_image" in output and "ReLU" in output, output
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["model_injections"]["image_encoder"] == "components:good", config
    print("TUI rejects mismatched encoder, cancels then confirms compatible encoder, and probes it")

with tempfile.TemporaryDirectory(prefix="maven-assembly-race-") as directory:
    project = Path(directory)
    (project / "components.py").write_text("from torch import nn\ndef good(): return nn.ReLU()\n")
    wrapper = project / "race-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if "--if-config-revision" in sys.argv:
    path = Path(".model-maven/project.json")
    config = json.loads(path.read_text())
    config["concurrent_marker"] = "preserve"
    path.write_text(json.dumps(config))
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    inputs = ("\nm\nimage=3,4,4;tabular=5\n3\ny\n2\n\n\n"
              "a\n\nimage_encoder\ncomponents:good\ny\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                            env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    assert "PROJECT_CONFIG_CONFLICT" in result.stdout, result.stdout
    assert "Last diagnosis" not in result.stdout.split("PROJECT_CONFIG_CONFLICT", 1)[1], result.stdout
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["concurrent_marker"] == "preserve" and not config["model_injections"], config
    print("Configuration changed after preview: conditional commit rejected and concurrent edit preserved")

with tempfile.TemporaryDirectory(prefix="maven-assembly-unknown-") as directory:
    project = Path(directory)
    (project / "components.py").write_text("from torch import nn\ndef good(): return nn.ReLU()\n")
    wrapper = project / "uncertain-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, subprocess, sys
if "--if-config-revision" in sys.argv:
    result = subprocess.run([{cli!r}] + sys.argv[1:], capture_output=True)
    if result.returncode == 0:
        print("invalid receipt")
        sys.exit(0)
    sys.stdout.buffer.write(result.stdout)
    sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    inputs = ("\nm\nimage=3,4,4;tabular=5\n3\ny\n2\n\n\n"
              "a\n\nimage_encoder\ncomponents:good\ny\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                            env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    assert "outcome is unknown" in result.stdout, result.stdout
    assert "Last diagnosis" not in result.stdout.split("outcome is unknown", 1)[1], result.stdout
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["model_injections"]["image_encoder"] == "components:good", config
    print("Lost commit receipt: TUI clears old diagnosis and requires checking saved state before retry")
