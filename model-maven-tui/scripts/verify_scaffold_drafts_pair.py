#!/usr/bin/env python3
"""Resume cancelled starter forms without mixing separate projects."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-scaffold-drafts-") as directory:
    project = Path(directory) / "first"
    other = Path(directory) / "second"
    project.mkdir(); other.mkdir()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(Path(directory) / "prefs.json"), COLUMNS="160")
    inputs = ("\nm\na=4;b=2;@a_encoder=b\nb\n"
              f"o\n{other}\nm\nx=7\nb\n"
              f"o\n{project}\nm\n\n3\nn\n"
              "m\n\n\ny\n2\n\n\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
                            input=inputs, capture_output=True, text=True, timeout=120, check=True)
    assert result.stdout.count("Custom model preview") == 2, result.stdout
    assert "Custom model project created." in result.stdout and "Observed calls for this batch" in result.stdout, result.stdout
    source = (project / "model_starter.py").read_text()
    assert '"a_encoder": "b"' in source and '"b_encoder": "b"' in source, source
    assert not (other / "model_starter.py").exists()
    assert "Current: a=4;b=2;@a_encoder=b" in result.stderr, result.stderr
    print("Cancelled starter form survives project switching and cancelled preview without mixing drafts")
