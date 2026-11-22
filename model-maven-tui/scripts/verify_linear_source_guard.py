#!/usr/bin/env python3
"""Change project source after the Linear diagnosis preview and before commit."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-linear-source-guard-") as directory:
    project = Path(directory)
    (project / "fixture.py").write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 8)
    def forward(self, x): return self.head(x)
def model(): return Model()
def batch(): return torch.ones(2, 48)
''')
    wrapper = project / "wrapper.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if "layer" in sys.argv and "--if-source-manifest" in sys.argv:
    path = Path("fixture.py")
    path.write_text(path.read_text() + "\\n# source changed after preview\\n")
os.execv({cli!r}, [{cli!r}, *sys.argv[1:]])
''')
    wrapper.chmod(0o700)
    env = os.environ.copy()
    env.update(MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project, env=env, input="\n1\nfixture:model\n2\n\nfixture:batch\n7\n48\ny\nq\n", capture_output=True, text=True, timeout=120, check=True)
    assert "Commit will compare the worker-recorded project source manifest" in result.stdout, result.stdout
    assert "SOURCE_MANIFEST_CONFLICT" in result.stdout, result.stdout
    assert not (project / ".model-maven/variants/working.json").exists()
    print("Linear repair refuses commit when recorded project source changes after preview")
