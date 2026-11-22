#!/usr/bin/env python3
"""Switch a dependent encoder output and fusion together through real TUI."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-atomic-encoder-fusion-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    (project / "components.py").write_text('''import torch
from torch import nn
class Native(nn.Module):
    def forward(self, values):
        x, tabular = values
        assert x.ndim == 3 and x.dtype == torch.int64
        return torch.cat([x.float().mean(1), tabular], dim=1)
class Flattened(nn.Module):
    def forward(self, values):
        x, tabular = values
        assert x.ndim == 2 and x.dtype == torch.float32
        return torch.cat([x, tabular], dim=1)
def native(): return Native()
def flat(): return Flattened()
def head(): return nn.Linear(6, 3)
''')
    def run(*args, success=True):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        assert (proc.returncode == 0) == success, report
        return report
    args = ("scaffold", "--input", "text:int64=?5,4", "--input", "tabular=2", "--output-dim", "3",
            "--native-encoder", "text_encoder", "--inject", "fusion=components:native", "--inject", "head=components:head")
    plan = run(*args)
    run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    config = project / ".model-maven/project.json"
    original = config.read_bytes()
    run("assembly", "encoder-output", "text_encoder", "flatten", "--dry-run", success=False)
    run("assembly", "set", "fusion", "--factory", "components:flat", "--dry-run", success=False)
    assert config.read_bytes() == original
    def session(confirm):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
            input=f"\na\npatch\nfusion=components:flat;@output.text_encoder=flatten\n\n{{}}\ny\n{confirm}\nq\n",
            capture_output=True, text=True, timeout=180, check=True).stdout
    cancelled = session("n")
    assert "Encoder output overrides" in cancelled and config.read_bytes() == original, cancelled
    saved = session("y")
    assert "Batch assembly preview" in saved and "Assembly saved" in saved, saved
    actual = json.loads(config.read_text())
    assert actual["model_injections"]["fusion"] == "components:flat", actual
    assert actual["model_params"]["encoder_output_modes"]["text_encoder"] == "flatten", actual
    run("model", "probe", "--batch-param", "text_dim1=7")
    run("try", "--step", "model_starter:step", "--steps", "1")
    generated = run("assembly", "generate")
    assert generated["data"]["generated_validation"]["forward"] == "passed", generated
    restored = subprocess.run(["python3", "-c", "from generated.model_assembly import build_model; import model_starter; assert build_model()(**model_starter.batch(text_dim1=7)['kwargs']).shape == (2,3)"], cwd=project, env=env, capture_output=True, text=True, timeout=60)
    assert restored.returncode == 0, (restored.stdout, restored.stderr)
    print("TUI atomically switches mutually incompatible encoder output and fusion; cancel, real forward, trial and reconstruction pass")
