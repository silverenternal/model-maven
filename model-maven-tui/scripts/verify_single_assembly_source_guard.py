#!/usr/bin/env python3
"""Exercise source changes between single-seam preview and commit through the TUI."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-single-source-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    (project / "components.py").write_text(
        "from torch import nn\ndef good(): return nn.ReLU()\n"
        "def other(): return nn.Identity()\n")
    wrapper = project / "race-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if "--if-source-manifest" in sys.argv:
    path = Path("model_starter.py")
    with path.open("a") as stream: stream.write("# changed after preview\\n")
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)

    def session(inputs, binary=cli):
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(binary)],
                                cwd=project, env=env, input=inputs, text=True,
                                capture_output=True, timeout=180, check=True)
        return result.stdout

    created = session("\nm\nimage=3,4,4;tabular=5\n3\ny\nq\n")
    assert "Custom model project created" in created, created
    def bindings():
        return json.loads((project / ".model-maven/project.json").read_text())["model_injections"]

    add = session("\na\n\nimage_encoder\ncomponents:good\ny\nq\n", wrapper)
    assert "SOURCE_MANIFEST_CONFLICT" in add and not bindings(), add
    success = session("\na\n\nimage_encoder\ncomponents:good\ny\nq\n")
    assert "Assembly saved" in success and bindings()["image_encoder"] == "components:good", success
    for action, inputs in [
        ("set", "\na\nset\nimage_encoder\ncomponents:other\ny\nq\n"),
        ("remove", "\na\nremove\nimage_encoder\ny\nq\n"),
    ]:
        output = session(inputs, wrapper)
        assert "SOURCE_MANIFEST_CONFLICT" in output, (action, output)
        assert bindings()["image_encoder"] == "components:good", (action, bindings())
    assert (project / "model_starter.py").read_text().count("# changed after preview") == 3
    print("Single-seam add, set and remove reject observed source changes; normal add succeeds")
