#!/usr/bin/env python3
"""Create an explicitly routed, shared-input model with real TUI and CLI."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-routed-starter-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
                            input="\nm\na=4;b=2;@a_encoder=b\n3\ny\n2\n\n\nq\n",
                            capture_output=True, text=True, timeout=120, check=True)
    assert "a_encoder ← b" in result.stdout and "b_encoder ← b" in result.stdout, result.stdout
    assert "Unconnected input: a" in result.stdout, result.stdout
    assert "Custom model project created." in result.stdout and "Observed calls for this batch" in result.stdout, result.stdout
    script = "import torch, model_starter as s\nm=s.model()\nassert m.head.in_features == 4\na=torch.zeros(2,4); b=torch.ones(2,2)\nassert m(a=a,b=b).shape == (2,3)\nassert torch.equal(m.encoders['enc_a'](b),m.encoders['enc_b'](b))\n"
    check = subprocess.run(["python3", "-c", script], cwd=project, env=env, capture_output=True, text=True, timeout=60)
    assert check.returncode == 0, (check.stdout, check.stderr)
    print("TUI previews explicit shared-input routes and generated model runs in a fresh process")
