#!/usr/bin/env python3
"""Check optional forward-input binding for single-seam TUI edits."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-single-input-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    wrapper = project / "race-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if "--if-input-content" in sys.argv:
    Path("input-value.txt").write_text("2")
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)

    def session(inputs, binary=cli):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(binary)],
                              cwd=project, env=env, input=inputs, text=True,
                              capture_output=True, timeout=180, check=True).stdout

    assert "Custom model project created" in session("\nm\nimage=3,4,4;tabular=5\n3\ny\nq\n")
    with (project / "model_starter.py").open("a") as stream:
        stream.write('''\nfrom pathlib import Path
_original_batch = batch
def batch(batch_size=2):
    result = _original_batch(batch_size=batch_size)
    result["kwargs"]["image"] *= float(Path("input-value.txt").read_text())
    return result
''')
    (project / "components.py").write_text("from torch import nn\ndef good(): return nn.ReLU()\ndef other(): return nn.Identity()\n")
    value = project / "input-value.txt"
    value.write_text("1")
    config = project / ".model-maven/project.json"
    def bindings():
        return json.loads(config.read_text())["model_injections"]
    def raced(inputs):
        value.write_text("1")
        return session(inputs, wrapper)

    add = raced("\na\n\nimage_encoder\ncomponents:good\ninput\nq\n")
    assert "INPUT_CONTENT_MISMATCH" in add and not bindings(), add
    value.write_text("1")
    stable = session("\na\n\nimage_encoder\ncomponents:good\ninput\nq\n")
    assert "Assembly saved" in stable and bindings()["image_encoder"] == "components:good", stable
    for inputs in ("\na\nset\nimage_encoder\ncomponents:other\ninput\nq\n",
                   "\na\nremove\nimage_encoder\ninput\nq\n"):
        output = raced(inputs)
        assert "INPUT_CONTENT_MISMATCH" in output and bindings()["image_encoder"] == "components:good", output
    value.write_text("1")
    unpinned = session("\na\nset\nimage_encoder\ncomponents:other\ny\nq\n", wrapper)
    assert "Assembly saved" in unpinned and bindings()["image_encoder"] == "components:other", unpinned
    cfg = json.loads(config.read_text())
    cfg.pop("batch", None)
    cfg.pop("batch_binding", None)
    cfg.pop("batch_params", None)
    config.write_text(json.dumps(cfg))
    missing = subprocess.run([cli, "--json", "--no-interactive", "--if-input-content", "0" * 64,
                              "assembly", "remove", "image_encoder"], cwd=project, env=env,
                             capture_output=True, text=True, timeout=90)
    report = json.loads(missing.stdout)
    assert missing.returncode != 0 and report["diagnostics"][0]["code"] == "INPUT_CONTENT_UNAVAILABLE", report
    assert bindings()["image_encoder"] == "components:other", report
    print("Optional same-input condition rejects add/set/remove races, accepts stable input and permits unpinned edits")
