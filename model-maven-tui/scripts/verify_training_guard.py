#!/usr/bin/env python3
"""Verify a changed project configuration cannot launch from a stale TUI plan."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-training-guard-") as directory:
    project = Path(directory)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    args = [cli, "--json", "--no-interactive", "scaffold", "--input", "left=4", "--input", "right=5", "--output-dim", "2"]
    preview = subprocess.run(args, cwd=project, env=env, capture_output=True, text=True, check=True)
    revision = json.loads(preview.stdout)["data"]["plan_revision"]
    subprocess.run(args + ["--apply", "--if-plan", revision], cwd=project, env=env, capture_output=True, check=True)
    wrapper = project / "race-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if "--if-config-revision" in sys.argv and ("try" in sys.argv or "train" in sys.argv):
    path = Path(".model-maven/project.json")
    config = json.loads(path.read_text())
    config["concurrent_marker"] = "preserve"
    path.write_text(json.dumps(config))
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                            env=env, input="\nr\ntry-wait\n\n\n\n\n\n\ny\nq\n", capture_output=True, text=True, timeout=120, check=True)
    assert "PROJECT_CONFIG_CONFLICT" in result.stdout, result.stdout
    assert not (project / ".model-maven/runs").exists()
    assert json.loads((project / ".model-maven/project.json").read_text())["concurrent_marker"] == "preserve"
    print("Training configuration changed after preview: launch refused before run creation")
