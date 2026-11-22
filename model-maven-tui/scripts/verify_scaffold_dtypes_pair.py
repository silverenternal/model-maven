#!/usr/bin/env python3
"""Create and assemble mixed-dtype inputs through the real TUI."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-dtypes-pair-") as directory:
    project = Path(directory)
    (project / "encoder.py").write_text('''import torch
from torch import nn
class TokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 1)
    def forward(self, x):
        assert x.dtype == torch.int64
        return self.embedding(x)
def make(): return TokenEncoder()
''')
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"), COLUMNS="160")
    inputs = ("\nm\nimage=3,4,4;tokens:int64=4;mask:bool=4\n3\ny\n"
              "a\n\ntokens_encoder\nencoder:make\ny\n2\n\n\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                            env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    output = result.stdout
    assert "Custom model project created." in output and "Assembly saved" in output, (output, result.stderr)
    assert "TokenEncoder" in output and "Observed calls for this batch" in output, output
    assert "Default fusion converts encoder outputs to float32" in output, output
    source = (project / "model_starter.py").read_text()
    assert "dtype=torch.int64" in source and "dtype=torch.bool" in source
    probe = subprocess.run([cli, "--json", "--no-interactive", "model", "probe"], cwd=project,
                           env=env, capture_output=True, text=True, timeout=90, check=True)
    assert json.loads(probe.stdout)["status"] == "success"
    print("TUI creates mixed-dtype model; real assembly preserves int64 at token encoder")
