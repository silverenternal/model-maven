#!/usr/bin/env python3
"""Check dynamic input snapshot replay and cleanup for single-seam edits."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-single-snapshot-") as directory:
    project = Path(directory) / "project"
    project.mkdir()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    def session(inputs, binary=cli):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(binary)],
                              cwd=project, env=env, input=inputs, text=True,
                              capture_output=True, timeout=180, check=True).stdout

    assert "Custom model project created" in session("\nm\nimage=3,4,4;tabular=5\n3\ny\nq\n")
    with (project / "model_starter.py").open("a") as stream:
        stream.write('''\nfrom pathlib import Path
_original_batch = batch
def batch(batch_size=2):
    path = Path("input-value.txt")
    value = float(path.read_text())
    path.write_text(str(value + 1))
    result = _original_batch(batch_size=batch_size)
    result["kwargs"]["image"] *= value
    return result
''')
    (project / "components.py").write_text("from torch import nn\ndef good(): return nn.ReLU()\ndef other(): return nn.Identity()\n")
    counter = project / "input-value.txt"
    counter.write_text("1")
    config = project / ".model-maven/project.json"
    def bindings(): return json.loads(config.read_text())["model_injections"]
    def cleaned(output):
        match = re.search(r"Temporary file: ([^\s]+)", output)
        assert match and not Path(match.group(1)).exists(), output
        assert "Temporary input snapshot removed" in output, output

    saved = session("\na\nadd+snapshot\nimage_encoder\ncomponents:good\ny\nq\n")
    assert "Assembly saved" in saved and bindings()["image_encoder"] == "components:good", saved
    assert float(counter.read_text()) == 2, saved
    cleaned(saved)
    cancelled = session("\na\nset+snapshot\nimage_encoder\ncomponents:other\nn\nq\n")
    assert bindings()["image_encoder"] == "components:good", cancelled
    assert float(counter.read_text()) == 3, cancelled
    cleaned(cancelled)

    wrapper = Path(directory) / "bad-preview.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, subprocess, sys
if "--capture-input-to" in sys.argv:
    result = subprocess.run([{cli!r}] + sys.argv[1:], capture_output=True)
    report = json.loads(result.stdout)
    if report.get("status") == "success": report["data"]["input_evidence"]["digest"] = None
    sys.stdout.write(json.dumps(report)); sys.stderr.buffer.write(result.stderr); sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    failed = session("\na\nremove+snapshot\nimage_encoder\nq\n", wrapper)
    assert "Input snapshot could not be captured and verified" in failed, failed
    assert bindings()["image_encoder"] == "components:good", failed
    assert float(counter.read_text()) == 4, failed
    cleaned(failed)
    for args in [
        ["--replay-input-from", "missing.snapshot", "assembly", "remove", "image_encoder"],
        ["--capture-input-to", "illegal.snapshot", "assembly", "remove", "image_encoder"],
    ]:
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, text=True, capture_output=True, timeout=90)
        report = json.loads(result.stdout)
        assert result.returncode != 0 and report["diagnostics"][0]["code"] == "INPUT_SNAPSHOT_UNSUPPORTED", report
        assert bindings()["image_encoder"] == "components:good", report
    print("Dynamic single-seam input captured once and replayed; save, cancel and bad preview clean up")
