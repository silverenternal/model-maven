#!/usr/bin/env python3
"""Exercise source ModuleList and ModuleDict edits across TUI restarts."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
kinds = [os.environ["MODEL_MAVEN_SOURCE_CONTAINER"]] if os.environ.get("MODEL_MAVEN_SOURCE_CONTAINER") else ["modulelist", "moduledict"]
for kind in kinds:
    assert kind in {"modulelist", "moduledict"}
    with tempfile.TemporaryDirectory(prefix="maven-source-container-") as directory:
        project = Path(directory)
        source = project / "fixture.py"
        source.write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity()])
        self.routes = nn.ModuleDict({"main": nn.Identity()})
        self.head = nn.Linear(4, 2)
    def forward(self, x):
        for block in self.blocks: x = block(x)
        return self.head(self.routes["main"](x))
def model(): return Model()
def batch(): return torch.ones(2, 4)
''')
        env = os.environ.copy()
        env.update(MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
        def run(*args):
            result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env, capture_output=True, text=True, check=True, timeout=120)
            return json.loads(result.stdout)
        run("init", "--model", "fixture:model", "--batch", "fixture:batch")
        def modules(): return run("layer", "list", "--model", "fixture:model")["data"]["modules"]
        def index(path): return next(i for i, item in enumerate(modules(), 1) if item["path"] == path)
        def exists(path): return any(item["path"] == path for item in modules())
        def session(lines):
            result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env, input="\nh\n" + lines + "\nq\n", capture_output=True, text=True, check=True, timeout=120)
            return result.stdout
        parent, child, name = ("model.blocks", "model.blocks.1", "") if kind == "modulelist" else ("model.routes", "model.routes.aux", "aux")
        original = source.read_bytes()
        def add(confirm):
            output = session(f"add\n{index(parent)}\nReLU\n{name}\n\n{{}}\n{confirm}")
            assert "Source preview" in output, output
            return output
        add("n")
        assert source.read_bytes() == original and not exists(child)
        output = add("apply")
        assert "Post-save verification" in output and '"status":"passed"' in output and exists(child), output
        output = session("undo\napply")
        assert "separate forward recheck" in output and not exists(child), output
        recheck = output.split("separate forward recheck", 1)[1].split("Model-Maven Workbench", 1)[0]
        assert "Completed" in recheck and "Failed" not in recheck, output
        output = session("redo\napply")
        assert "separate forward recheck" in output and exists(child), output
        recheck = output.split("separate forward recheck", 1)[1].split("Model-Maven Workbench", 1)[0]
        assert "Completed" in recheck and "Failed" not in recheck, output
        output = session(f'set\n{index(child)}\n{{"inplace":true}}\napply')
        assert "Post-save verification" in output and '"status":"passed"' in output, output
        assert next(item for item in modules() if item["path"] == child)["editable_params"]["inplace"] is True
        output = session(f"replace\n{index(child)}\nGELU\n{{}}\napply")
        assert "Post-save verification" in output and '"status":"passed"' in output, output
        assert next(item for item in modules() if item["path"] == child)["type"] == "GELU"
        output = session(f"remove\n{index(child)}\napply")
        assert "Post-save verification" in output and '"status":"passed"' in output and not exists(child), output
        print(f"{kind}: source preview, cancel, add, set, replace, undo, redo and remove persist across TUI restarts")
