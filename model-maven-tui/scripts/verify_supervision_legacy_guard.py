#!/usr/bin/env python3
"""A CLI without rule-state revisions cannot start confirmed supervised training."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-supervision-legacy-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), REAL_MAVEN_CLI=cli, COLUMNS="160")
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, check=True, timeout=60)
        return json.loads(result.stdout)
    args = ("scaffold", "--input", "x=4", "--output-dim", "2")
    preview = run(*args)
    run(*args, "--apply", "--if-plan", preview["data"]["plan_revision"])
    wrapper = project / "legacy-cli"
    wrapper.write_text('''#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
real = os.environ['REAL_MAVEN_CLI']
if 'try' in args or 'train' in args:
    open('unexpected-start', 'w').close()
if 'supervise' in args and 'list' in args:
    result = subprocess.run([real] + args, capture_output=True, text=True)
    report = json.loads(result.stdout)
    report['data'].pop('state_revision', None)
    print(json.dumps(report)); sys.stderr.write(result.stderr); sys.exit(result.returncode)
os.execv(real, [real] + args)
''')
    wrapper.chmod(0o755)
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                            env=env, input="\nr\ntry-wait\n\n\n\n\n\ny\nq\n", text=True,
                            capture_output=True, check=True, timeout=90)
    assert "does not provide a supervision revision" in result.stdout, result.stdout
    assert not (project / "unexpected-start").exists()
    print("Legacy revision-less CLI is rejected before supervised training starts")
