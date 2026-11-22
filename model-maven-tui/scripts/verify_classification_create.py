#!/usr/bin/env python3
"""Exercise classification setup through the real TUI, then run a real forward."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-create-pair-") as directory:
    project = Path(directory)
    for name, color in [("cats", "red"), ("dogs", "blue")]:
        path = project / "data" / name
        path.mkdir(parents=True)
        Image.new("RGB", (16, 16), color).save(path / "sample.png")
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    cancelled = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli],
                               cwd=project, env=env, input="\nb\n\ndata\n\nq\n",
                               capture_output=True, text=True, timeout=120, check=True)
    assert "Classification preview" in cancelled.stdout, cancelled.stdout
    assert not (project / "classification.py").exists()
    assert not (project / ".model-maven/project.json").exists()
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli],
                            cwd=project, env=env, input="\nb\n\ndata\ny\n2\n\n\nq\n",
                            capture_output=True, text=True, timeout=120, check=True)
    assert "Classification preview" in result.stdout, result.stdout
    assert "cats, dogs" in result.stdout, result.stdout
    assert "Classification project created." in result.stdout, result.stdout
    assert "Batch  classification:batch" in result.stdout, result.stdout
    assert "Observed calls for this batch" in result.stdout and "model.7  #1  Linear" in result.stdout, result.stdout
    assert (project / "classification.py").is_file(), result.stdout
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["model"] == "classification:model" and config["batch"] == "classification:batch"
    print("TUI creates Tiny CNN with two real images and displays its actual forward calls")

with tempfile.TemporaryDirectory(prefix="maven-create-conflict-") as directory:
    project = Path(directory)
    for name in ("cats", "dogs"):
        folder = project / "data" / name
        folder.mkdir(parents=True)
        Image.new("RGB", (16, 16), "red").save(folder / "sample.png")
    source = project / "classification.py"
    source.write_text("# Existing user file\n")
    original = source.read_bytes()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"))
    conflict = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli],
                              cwd=project, env=env, input="\nb\n\ndata\nq\n",
                              capture_output=True, text=True, timeout=120, check=True)
    assert "CREATE_TARGET_CONFLICT" in conflict.stdout and "classification.py" in conflict.stdout, conflict.stdout
    assert source.read_bytes() == original and not (project / ".model-maven/project.json").exists()
    print("Existing classification source is identified and preserved during TUI creation")

if os.name == "posix" and os.geteuid() != 0:
    with tempfile.TemporaryDirectory(prefix="maven-create-init-failure-") as directory:
        project = Path(directory)
        for name in ("cats", "dogs"):
            folder = project / "data" / name
            folder.mkdir(parents=True)
            Image.new("RGB", (16, 16), "red").save(folder / "sample.png")
        metadata = project / ".model-maven"
        metadata.mkdir()
        sentinel = metadata / "keep.txt"
        sentinel.write_text("user content\n")
        metadata.chmod(0o555)
        try:
            env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"))
            failure = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                env=env, input="\nb\n\ndata\ny\nq\n", capture_output=True, text=True, timeout=120, check=True)
            assert "PROJECT_CONFIG_WRITE_FAILED" in failure.stdout, failure.stdout
            assert ".model-maven" in failure.stdout and "Model-Maven Workbench" in failure.stdout, failure.stdout
            assert not (project / "classification.py").exists() and not (metadata / "project.json").exists()
            assert sentinel.read_text() == "user content\n"
        finally:
            metadata.chmod(0o755)
        print("Failed initialization removes generated starter and preserves user files")
else:
    print("Initialization permission-failure fixture skipped: requires non-root POSIX user")

with tempfile.TemporaryDirectory(prefix="maven-create-malformed-") as directory:
    project = Path(directory)
    wrapper = project / "cli-wrapper.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, sys
if "create" in sys.argv:
    print('{{"protocol_version":1,"operation":"create","status":"success","data":{{"committed":false}}}}')
    sys.exit(0)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)],
                            cwd=project, env=env, input="\nb\n\ndata\nq\n",
                            capture_output=True, text=True, timeout=60, check=True)
    assert "classification preview has no revision" in result.stdout, result.stdout
    assert not (project / "classification.py").exists()
    print("Malformed preview remains a recoverable protocol error; no files created")
