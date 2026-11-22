#!/usr/bin/env python3
"""Edit real source or variant layers with a registered component Schema."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
targets = [os.environ["MODEL_MAVEN_FORM_TARGET"]] if os.environ.get("MODEL_MAVEN_FORM_TARGET") else ["variant", "source"]
for target in targets:
    assert target in {"source", "variant"}
    with tempfile.TemporaryDirectory(prefix="maven-component-form-") as directory:
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
        (project / "custom_layers.py").write_text('''from torch import nn
def make(negative_slope=0.1, tag="small"):
    return nn.LeakyReLU(negative_slope=negative_slope)
''')
        env = os.environ.copy()
        env.update(MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
        def run(*args):
            result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env, capture_output=True, text=True, timeout=120, check=True)
            return json.loads(result.stdout)
        run("init", "--model", "fixture:model", "--batch", "fixture:batch")
        schema = {"type":"object", "properties":{"negative_slope":{"type":"number", "minimum":0, "maximum":1, "default":0.1}, "tag":{"type":"string", "enum":["small", "big"]}}, "required":["tag"], "additionalProperties":False}
        run("component", "add", "demo/activation", "--kind", "layer", "--version", "1.2.3", "--entrypoint", "custom_layers:make", "--params-schema", json.dumps(schema))
        modules = run("layer", "list", "--model", "fixture:model")["data"]["modules"]
        index = next(i for i, item in enumerate(modules, 1) if item["path"] == "model.seq.activation")
        if target == "variant":
            # Bad bound and enum values must be rejected locally; then preview and save.
            if os.environ.get("MODEL_MAVEN_FORM_DEFAULT") == "1":
                lines = f'\nw\n{index}\nreplace\ncomponent:demo/activation\n\n\nsmall\ny\nq\n'
            else:
                lines = f'\nw\n{index}\nreplace\ncomponent:demo/activation\n\n2\n0.25\nbad\nsmall\ny\nq\n'
        else:
            # JSON remains available for nested or advanced parameters.
            lines = f'\nh\nreplace\n{index}\ncomponent:demo/activation\njson\n{{"negative_slope":0.4,"tag":"big"}}\napply\nq\n'
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env, input=lines, capture_output=True, text=True, timeout=120, check=True)
        output = result.stdout
        assert "Component parameter Schema" in output and "demo/activation@1.2.3" in output, output
        if target == "variant":
            assert "independent recheck" in output, output
            if os.environ.get("MODEL_MAVEN_FORM_DEFAULT") != "1":
                assert "above the maximum" in output and "not in the allowed choices" in output, output
            after = run("layer", "list", "--variant", "working")["data"]["modules"]
            edit = run("variant", "show", "working")["data"]["variant"]["edits"][-1]
            expected_slope = 0.1 if os.environ.get("MODEL_MAVEN_FORM_DEFAULT") == "1" else 0.25
            assert edit["component_ref"]["version"] == "1.2.3" and edit["params"] == {"negative_slope":expected_slope, "tag":"small"}, edit
        else:
            assert "Post-save verification" in output and '"status":"passed"' in output, output
            after = run("layer", "list", "--model", "fixture:model")["data"]["modules"]
            source = (project / "fixture.py").read_text()
            assert "0.4" in source and "big" in source, source
            value = subprocess.check_output(["python3", "-c", "import fixture; print(fixture.model().seq.activation.negative_slope)"], cwd=project, env=env, text=True, timeout=60)
            assert float(value.strip()) == 0.4, value
        assert next(item for item in after if item["path"] == "model.seq.activation")["type"] == "LeakyReLU", after
        print(f"{target}: registered component Schema form resolves exact version and safely edits model")
