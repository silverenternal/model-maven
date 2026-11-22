#!/usr/bin/env python3
"""Exercise an actual historical native CLI; pass its binary as the first argument."""
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
if len(sys.argv) != 2:
    raise SystemExit("usage: verify_legacy_cli_pair.py /path/to/historical/m-model")
legacy_source = Path(sys.argv[1]).resolve(strict=True)
with tempfile.TemporaryDirectory(prefix="maven-real-legacy-") as directory:
    base = Path(directory)
    old = base / "legacy cli with spaces"
    shutil.copy2(legacy_source, old)
    project = base / "project"
    project.mkdir()
    source = project / "fixture.py"
    source.write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)
    def forward(self, x): return self.head(x)
def model(): return Model()
def batch(): return torch.ones(2, 4)
''')
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    init = subprocess.run([old, "--json", "init", "--model", "fixture:model", "--batch", "fixture:batch"],
                          cwd=project, env=env, text=True, capture_output=True, timeout=90, check=True)
    assert json.loads(init.stdout)["status"] == "success"
    before = source.read_bytes()
    before_config = (project / ".model-maven/project.json").read_bytes()
    def session(executable, inputs, environment=env):
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(executable)],
                                cwd=project, env=environment, input=inputs, text=True,
                                capture_output=True, timeout=180, check=True)
        return result.stdout
    output = session(old, "\n1\n\n2\n\n\n3\nt\nq\n")
    assert "model.head" in output and "Observed calls for this batch" in output, output
    assert "Historical CLI detected" in output, output
    assert "Historical CLI cannot edit supervision" in output, output
    assert source.read_bytes() == before and (project / ".model-maven/project.json").read_bytes() == before_config
    path_env = dict(env, PATH=str(base) + os.pathsep + env.get("PATH", ""))
    path_output = session(old.name, "\n3\nq\n", path_env)
    assert "Python" in path_output and "Completed" in path_output, path_output
    upgrade_source = os.environ.get("MODEL_MAVEN_UPGRADE_CLI")
    upgrade_output = None
    downgrade_output = None
    if upgrade_source:
        process = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", str(old)],
                                   cwd=project, env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.stdin.write(b"\n")
        process.stdin.flush()
        prefix = b""
        deadline = time.monotonic() + 30
        while b"Historical CLI detected" not in prefix:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                process.kill()
                process.communicate()
                raise AssertionError("Timed out waiting for historical CLI notice")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise AssertionError("TUI exited before historical CLI notice")
            prefix += chunk
        replacement = base / "new cli"
        shutil.copy2(upgrade_source, replacement)
        os.replace(replacement, old)
        rest, errors = process.communicate(b"c\nr\n\nq\n", timeout=90)
        upgrade_output = (prefix + rest).decode()
        assert process.returncode == 0, errors.decode()
        assert "Project status unavailable" in upgrade_output, upgrade_output
        assert "Project and CLI status" in upgrade_output, upgrade_output
        assert "Project configured" in upgrade_output and "a. Assemble components" in upgrade_output, upgrade_output
        assert source.read_bytes() == before and (project / ".model-maven/project.json").read_bytes() == before_config
        process = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", str(old)],
                                   cwd=project, env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.stdin.write(b"\n")
        process.stdin.flush()
        prefix = b""
        deadline = time.monotonic() + 30
        while b"Project configured" not in prefix:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                process.kill()
                process.communicate()
                raise AssertionError("Timed out waiting for current CLI project status")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise AssertionError("TUI exited before current CLI project status")
            prefix += chunk
        shutil.copy2(legacy_source, replacement)
        os.replace(replacement, old)
        rest, errors = process.communicate(b"c\nr\n\nq\n", timeout=90)
        downgrade_output = (prefix + rest).decode()
        assert process.returncode == 0, errors.decode()
        assert "Legacy CLI: this action is unavailable" in downgrade_output, downgrade_output
        assert source.read_bytes() == before and (project / ".model-maven/project.json").read_bytes() == before_config
    evidence = os.environ.get("MODEL_MAVEN_LEGACY_EVIDENCE")
    if evidence:
        evidence_path = Path(evidence)
        evidence_path.mkdir(parents=True, exist_ok=True)
        (evidence_path / "session.txt").write_text(output)
        (evidence_path / "path-session.txt").write_text(path_output)
        (evidence_path / "cli-sha256.txt").write_text(hashlib.sha256(legacy_source.read_bytes()).hexdigest() + "\n")
        if upgrade_output is not None:
            (evidence_path / "upgrade-session.txt").write_text(upgrade_output)
            (evidence_path / "downgrade-session.txt").write_text(downgrade_output)
    print("Actual historical CLI SHA-256", hashlib.sha256(legacy_source.read_bytes()).hexdigest())
    print("Historical native CLI: inspect, probe and doctor work; guarded edit refuses without changes; PATH and spaces work")
