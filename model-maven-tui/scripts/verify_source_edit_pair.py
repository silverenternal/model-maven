#!/usr/bin/env python3
"""Exercise guarded source editing, concurrency, undo and redo in the real TUI."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-source-edit-pair-") as directory:
    project = Path(directory)
    source = project / "fixture.py"
    source.write_text('''import torch
from torch import nn
from collections import OrderedDict
class Scale(nn.Module):
    def __init__(self, factor=1.0, offset=0.0, scale=1.0):
        super().__init__()
        self.factor, self.offset, self.scale = factor, offset, scale
    def forward(self, x): return x * self.factor * self.scale + self.offset
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(OrderedDict([('first', nn.ReLU())]))
        self.head = nn.Linear(4, 2)
        self.custom = Scale(
            # Keep this comment, including its comma.
        )
    def forward(self, x): return self.head(self.custom(self.seq(x)))
def model(): return Model()
def batch(): return torch.ones(2, 4)
def step(model, batch, context): return {"loss": model(batch).square().mean()}
''')
    wrapper = project / "wrapper.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if "--if-source-plan" in sys.argv and os.environ.get("SOURCE_EDIT_TEST_RACE") == "1":
    path = Path("fixture.py")
    path.write_text(path.read_text() + "\\n# concurrent edit\\n")
os.execv({cli!r}, [{cli!r}, *sys.argv[1:]])
''')
    wrapper.chmod(0o700)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "prefs.json")
    env["COLUMNS"] = "160"
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env, check=True, capture_output=True, text=True, timeout=120)
        return json.loads(result.stdout)
    run("init", "--model", "fixture:model", "--batch", "fixture:batch")
    trained = run("train", "--step", "fixture:step", "--steps", "1")
    trained_id = trained["data"]["run_id"]
    checkpoint = run("runs", "show", trained_id)["data"]["checkpoint"]
    checkpoint_file = project / ".model-maven/runs" / trained_id / "checkpoints" / checkpoint["file"]
    checkpoint_hash = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    assert checkpoint_hash == checkpoint["sha256"]
    modules = run("layer", "list", "--model", "fixture:model")["data"]["modules"]
    index = next(i for i, item in enumerate(modules, 1) if item["path"] == "model.head")
    assert modules[index - 1]["can_source_set"] is True, modules[index - 1]
    def session(lines, race=False):
        local = env.copy()
        if race: local["SOURCE_EDIT_TEST_RACE"] = "1"
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project, env=local, input="\nh\n" + lines + "\nq\n", capture_output=True, text=True, timeout=120, check=True).stdout
    original = source.read_bytes()
    output = session(f'set\n{index}\n{{"bias":false}}\nn')
    assert "Source preview" in output and source.read_bytes() == original, output
    compact = output.replace("\r", "").replace("\n", "")
    assert 'Previous module state not copied · model.head: ["bias"]' in compact, output
    assert 'Candidate copied state · model.head: ["weight"]' in compact, output
    assert "not a saved training checkpoint" in compact, output
    output = session(f'set\n{index}\n{{"bias":false}}\napply', race=True)
    assert "SOURCE_PLAN_CONFLICT" in output and "bias=False" not in source.read_text(), output
    baseline = source.read_bytes()
    assert b"# concurrent edit" in baseline
    output = session(f'set\n{index}\n{{"bias":false}}\napply')
    assert "Post-save verification" in output and '"status":"passed"' in output, output
    assert "bias=False" in source.read_text()
    output = session("undo\napply")
    assert "separate forward recheck" in output and source.read_bytes() == baseline, output
    recheck = output.split("separate forward recheck", 1)[1].split("Model-Maven Workbench", 1)[0]
    assert "Completed" in recheck and "Failed" not in recheck, output
    output = session("redo\napply")
    assert "separate forward recheck" in output and "bias=False" in source.read_text(), output
    recheck = output.split("separate forward recheck", 1)[1].split("Model-Maven Workbench", 1)[0]
    assert "Completed" in recheck and "Failed" not in recheck, output
    def layer_index(path):
        modules = run("layer", "list", "--model", "fixture:model")["data"]["modules"]
        return next(i for i, item in enumerate(modules, 1) if item["path"] == path)
    custom = layer_index("model.custom")
    output = session(f'set\n{custom}\n{{"offset":3.0,"scale":4.0}}\napply')
    assert "Post-save verification" in output and '"status":"passed"' in output, output
    assert "offset=3.0," in source.read_text() and "scale=4.0," in source.read_text(), source.read_text()
    assert "# Keep this comment, including its comma." in source.read_text(), source.read_text()
    subprocess.run(["python3", "-c", "import fixture; m=fixture.model(); assert (m.custom.offset, m.custom.scale) == (3.0, 4.0); m(fixture.batch())"], cwd=project, env=env, check=True, timeout=30)
    parent = layer_index("model.seq")
    output = session(f"add\n{parent}\nReLU\nextra\n\n{{}}\napply")
    assert "Post-save verification" in output and "earlier constructors can remain and still execute" in output, output
    added = layer_index("model.seq.extra")
    output = session(f"replace\n{added}\nGELU\n{{}}\napply")
    assert "Post-save verification" in output and "nn.GELU" in source.read_text(), output
    added = layer_index("model.seq.extra")
    output = session(f"remove\n{added}\napply")
    assert "Post-save verification" in output, output
    assert not any(item["path"] == "model.seq.extra" for item in run("layer", "list", "--model", "fixture:model")["data"]["modules"])
    assert hashlib.sha256(checkpoint_file.read_bytes()).hexdigest() == checkpoint_hash
    print("Source CRUD, candidate state effects and unchanged saved checkpoint verified")
