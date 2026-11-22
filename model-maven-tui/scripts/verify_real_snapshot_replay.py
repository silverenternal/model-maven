#!/usr/bin/env python3
"""Exercise opt-in session input replay with a real CLI in isolated projects."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))

artifact_root = os.environ.get("MODEL_MAVEN_INTEGRATION_ARTIFACTS_DIR")
for case_index, (factory, commit) in enumerate([("changing", True), ("changing", False), ("unsupported", False)]):
    with tempfile.TemporaryDirectory(prefix="maven-replay-tui-") as directory:
        project = Path(directory) / "project"
        project.mkdir()
        scratch = Path(directory) / "tmp"
        scratch.mkdir()
        (project / "fixture.py").write_text('''import torch
from torch import nn
from pathlib import Path
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 8)
    def forward(self, x): return self.head(x)
def model(): return Model()
def changing():
    path = Path('counter.txt')
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    return torch.full((2, 48), float(count))
def unsupported(): return torch.ones(48, 2).t()
''', encoding="utf-8")
        env = os.environ.copy()
        env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
        env["TMPDIR"] = str(scratch)
        inputs = f"\n1\nfixture:model\n2\n\n:snapshot\non\nfixture:{factory}\n"
        if commit:
            inputs += "7\n48\ny\n0\n"
        inputs += "q\n"
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli],
                                cwd=project, env=env, input=inputs, capture_output=True,
                                text=True, timeout=180, check=True)
        output = result.stdout
        assert not list(scratch.glob("model-maven-input-*")), (list(scratch.iterdir()), output)
        variant = project / ".model-maven/variants/working.json"
        assert variant.exists() == commit, output
        if factory == "unsupported":
            assert "Input snapshot unsupported" in output, output
        else:
            assert "CLI reports input snapshot captured" in output, output
            assert (project / "counter.txt").read_text() == "1", output
        if commit:
            assert "Bound input content matches diagnosis" in output, output
            assert "Guarded recheck · captured input replay" in output, output
            recheck = output.split("Guarded recheck · captured input replay", 1)[1].split("Model-Maven Workbench", 1)[0]
            assert "Completed" in recheck and "CLI reports captured input replayed" in recheck, output
            assert "Session receipt · historical result" in output, output
        if artifact_root:
            evidence = Path(artifact_root) / f"{case_index:02}-{factory}"
            evidence.mkdir(parents=True, exist_ok=False)
            (evidence / "fixture.py").write_text((project / "fixture.py").read_text(), encoding="utf-8")
            (evidence / "tui-stdout.txt").write_text(result.stdout, encoding="utf-8")
            (evidence / "tui-stderr.txt").write_text(result.stderr, encoding="utf-8")
            if variant.is_file():
                (evidence / "variant.json").write_bytes(variant.read_bytes())
            (evidence / "case.json").write_text(json.dumps({"batch_factory":factory,"committed":commit,"snapshot_cleaned":True}, indent=2) + "\n")
        print(f"{factory}: {'committed and replayed' if commit else 'quit without commit'}; temporary snapshot cleaned")

with tempfile.TemporaryDirectory(prefix="maven-old-cli-") as directory:
    project = Path(directory)
    wrapper = project / "old-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, sys
if "--describe-interface" in sys.argv:
    print('{{"protocol_version":1,"operation":"cli","status":"error","data":{{}},"diagnostics":[{{"code":"CLI_ARGUMENT_INVALID","message":"unknown argument"}}]}}')
    sys.exit(2)
if sys.argv[1:] == ["--help"]:
    print("Legacy CLI without snapshot options")
    sys.exit(0)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''', encoding="utf-8")
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)],
                            cwd=project, env=env,
                            input="\n2\nfixture:model\n:snapshot\non\nfixture:batch\nq\n",
                            capture_output=True, text=True, timeout=60, check=True).stdout
    assert "CLI does not advertise input snapshot capture and replay" in output, output
    assert not (project / ".model-maven/variants/working.json").exists()
    print("legacy CLI: snapshot explicitly unavailable; no fallback probe or write")

with tempfile.TemporaryDirectory(prefix="maven-snapshot-unconfirmed-") as directory:
    project = Path(directory) / "project"
    project.mkdir()
    scratch = Path(directory) / "tmp"
    scratch.mkdir()
    (project / "fixture.py").write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 8)
    def forward(self, x): return self.head(x)
def model(): return Model()
def batch(): return torch.ones(2, 48)
''', encoding="utf-8")
    wrapper = project / "cli-wrapper.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, subprocess, sys
if "layer" in sys.argv and "--dry-run" in sys.argv and "--replay-input-from" in sys.argv:
    result = subprocess.run([{cli!r}] + sys.argv[1:], capture_output=True)
    report = json.loads(result.stdout)
    report.get("data", {{}}).get("validation", {{}}).pop("batch_snapshot", None)
    print(json.dumps(report))
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''', encoding="utf-8")
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env.update(TMPDIR=str(scratch), MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"))
    output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)],
                            cwd=project, env=env,
                            input="\n1\nfixture:model\n2\n\n:snapshot\non\nfixture:batch\n7\n48\nq\n",
                            capture_output=True, text=True, timeout=120, check=True).stdout
    assert "Captured input replay unconfirmed; commit unavailable" in output, output
    assert "Commit guarded variant?" not in output, output
    assert not (project / ".model-maven/variants/working.json").exists()
    assert not list(scratch.glob("model-maven-input-*"))
    print("unconfirmed candidate replay: commit unavailable; snapshot cleaned")

with tempfile.TemporaryDirectory(prefix="maven-snapshot-timeout-") as directory:
    project = Path(directory) / "project"
    project.mkdir()
    scratch = Path(directory) / "tmp"
    scratch.mkdir()
    (project / "fixture.py").write_text('''import time
import torch
from torch import nn
def model(): return nn.Linear(32, 8)
def slow():
    time.sleep(20)
    return torch.ones(2, 48)
''', encoding="utf-8")
    env = os.environ.copy()
    env.update(TMPDIR=str(scratch), MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"),
               MODEL_MAVEN_TUI_CLI_TIMEOUT_MS="4000")
    output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli],
                            cwd=project, env=env,
                            input="\n2\nfixture:model\n:snapshot\non\nfixture:slow\nq\n",
                            capture_output=True, text=True, timeout=45, check=True).stdout
    assert "CLI operation timed out" in output, output
    assert not list(scratch.glob("model-maven-input-*")), list(scratch.iterdir())
    print("timed-out diagnostic: temporary snapshot directory cleaned")
