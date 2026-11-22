#!/usr/bin/env python3
"""Use backend Layer catalog forms with real source and variant edits."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
targets = [os.environ["MODEL_MAVEN_BUILTIN_TARGET"]] if os.environ.get("MODEL_MAVEN_BUILTIN_TARGET") else ["variant", "source"]
for target in targets:
    assert target in {"source", "variant"}
    with tempfile.TemporaryDirectory(prefix="maven-builtin-form-") as directory:
        project = Path(directory)
        (project / "fixture.py").write_text('''import torch
from collections import OrderedDict
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(OrderedDict([("activation", nn.ReLU())]))
        self.head = nn.Linear(4, 2)
    def forward(self, x): return self.head(self.seq(x))
def model(): return Model()
def batch(): return torch.ones(2, 4)
''')
        env = os.environ.copy()
        env.update(MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
        def run(*args):
            result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env, capture_output=True, text=True, timeout=120, check=True)
            return json.loads(result.stdout)
        run("init", "--model", "fixture:model", "--batch", "fixture:batch")
        modules = run("layer", "list", "--model", "fixture:model")["data"]["modules"]
        index = next(i for i, item in enumerate(modules, 1) if item["path"] == "model.seq.activation")
        if target == "variant":
            lines = f'\nw\n{index}\nreplace\nLinear\n:form\noops\n4\n4\n\ny\nq\n'
        else:
            lines = f'\nh\nreplace\n{index}\nDropout\n:form\n1.5\n0.25\n\napply\nq\n'
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env, input=lines, capture_output=True, text=True, timeout=120, check=True)
        output = result.stdout
        assert "Built-in Layer fields" in output, output
        if target == "variant":
            assert "Invalid JSON value" in output and "independent recheck" in output, output
            after = run("layer", "list", "--variant", "working")["data"]["modules"]
            layer = next(item for item in after if item["path"] == "model.seq.activation")
            assert layer["type"] == "Linear" and layer["editable_params"]["in_features"] == 4 and layer["editable_params"]["out_features"] == 4 and layer["editable_params"]["bias"] is True, layer
        else:
            assert "above the maximum" in output and "Post-save verification" in output, output
            after = run("layer", "list", "--model", "fixture:model")["data"]["modules"]
            layer = next(item for item in after if item["path"] == "model.seq.activation")
            assert layer["type"] == "Dropout" and layer["editable_params"]["p"] == 0.25 and layer["editable_params"]["inplace"] is False, layer
        print(f"{target}: catalog-guided built-in Layer parameters applied and forward verified")
