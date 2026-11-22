#!/usr/bin/env python3
"""Check real CLI/TUI input comparison in an isolated project."""

import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get(
    "MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model")
)

with tempfile.TemporaryDirectory(prefix="maven-input-content-") as directory:
    project = Path(directory)
    (project / "fixture.py").write_text(
        """import torch
from torch import nn
from pathlib import Path
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 8)
    def forward(self, x): return self.head(x)
def model(): return Model()
def stable(): return torch.ones(2, 48)
def changing():
    path = Path('counter.txt')
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    return torch.full((2, 48), float(count))
def unsupported(): return torch.ones(48, 2).t()
""",
        encoding="utf-8",
    )
    for factory, expected in [
        ("stable", "Bound input content matches diagnosis"),
        ("changing", "Bound input content differs from diagnosis"),
        ("unsupported", "Bound input content cannot be compared with diagnosis"),
    ]:
        env = os.environ.copy()
        env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / f"prefs-{factory}.json")
        result = subprocess.run(
            [tui, "--line-mode", "--lang", "en", "--cli", cli],
            cwd=project,
            env=env,
            input=f"\n1\nfixture:model\n2\n\nfixture:{factory}\n7\n48\nq\n",
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        assert expected in result.stdout, result.stdout
        assert "Candidate forward  passed" in result.stdout, result.stdout
        assert not (project / ".model-maven/variants/working.json").exists()
        print(f"{factory}: {expected}; preview did not write a variant")
