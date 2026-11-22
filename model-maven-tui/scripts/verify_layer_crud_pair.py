#!/usr/bin/env python3
"""Exercise variant layer CRUD through the real TUI and PyTorch CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-layer-crud-") as directory:
    project = Path(directory)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    (project / "fixture.py").write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.items = nn.ModuleList([nn.Identity()])
        self.routes = nn.ModuleDict({"main": nn.Identity()})
        self.head = nn.Linear(4, 2)
    def forward(self, x): return self.head(self.routes["main"](self.items[0](self.seq(x))))
def model(): return Model()
def batch(): return torch.ones(2, 4)
''')
    def run(*args):
        output = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120)
        result = json.loads(output.stdout)
        assert output.returncode == 0, result
        return result
    run("init", "--model", "fixture:model", "--batch", "fixture:batch")
    def edit(path, answers, committed):
        args = ["layer", "list"]
        if (project / ".model-maven/variants/working.json").exists(): args += ["--variant", "working"]
        modules = run(*args)["data"]["modules"]
        index = next(i for i, item in enumerate(modules, 1) if item["path"] == path)
        output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                                env=env, input=f"\nw\n{index}\n{answers}\nq\n", capture_output=True,
                                text=True, timeout=120, check=True).stdout
        assert "Layer preview" in output, output
        if committed:
            assert "Saved edit" in output and "independent recheck" in output, output
            recheck = output.split("independent recheck", 1)[1].split("Model-Maven Workbench", 1)[0]
            assert "Completed" in recheck and "Failed" not in recheck, output
        return output
    if os.environ.get("MODEL_MAVEN_LAYER_SET_ONLY") == "1":
        edit("model.head", 'set\n{"bias":false}\ny', True)
        modules = run("layer", "list", "--variant", "working")["data"]["modules"]
        assert next(m for m in modules if m["path"] == "model.head")["editable_params"]["bias"] is False
        print("Line-mode direct JSON parameter editing remains available")
        sys.exit(0)
    edit("model.seq.1", "replace\nGELU\n{}\n", False)
    assert not (project / ".model-maven/variants/working.json").exists()
    edit("model.seq.1", "replace\nGELU\n{}\ny", True)
    if os.environ.get("MODEL_MAVEN_LAYER_RECHECK_ONLY") == "1":
        print("TUI confirms a successful independent recheck after commit")
        sys.exit(0)
    edit("model.items", "add\nIdentity\n\n\n{}\ny", True)
    edit("model.routes", "add\nIdentity\naux\n\n{}\ny", True)
    edit("model.routes.aux", "remove\ny", True)
    edit("model.head", 'set\n{"bias":false}\ny', True)
    edit("model.head", "undo\ny", True)
    assert next(m for m in run("layer", "list", "--variant", "working")["data"]["modules"] if m["path"] == "model.head")["editable_params"]["bias"] is True
    edit("model.head", "redo\ny", True)
    modules = run("layer", "list", "--variant", "working")["data"]["modules"]
    assert next(m for m in modules if m["path"] == "model.seq.1")["type"] == "GELU"
    assert any(m["path"] == "model.items.1" for m in modules)
    assert not any(m["path"] == "model.routes.aux" for m in modules)
    assert next(m for m in modules if m["path"] == "model.head")["editable_params"]["bias"] is False
    assert "nn.ReLU()" in (project / "fixture.py").read_text()
    print("TUI previews, cancels, commits and rechecks edits across Sequential, ModuleList and ModuleDict")
