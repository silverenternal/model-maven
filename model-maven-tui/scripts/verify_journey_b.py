#!/usr/bin/env python3
"""Existing project: diagnose, reject stale/legacy edits, repair source and restart."""
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-journey-b-") as directory:
    project = Path(directory) / "existing"
    project.mkdir()
    source = project / "fixture.py"
    source.write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(in_features=32, out_features=8)
    def forward(self, x): return self.head(x)
def model(): return Model()
def batch(scale): return torch.full((2, 48), scale, dtype=torch.float32)
''')
    (project / "unrelated.txt").write_text("user-owned content\n")
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    trace_path = project.parent / "tui-cli-receipts.jsonl"
    trace = project.parent / "trace-cli.py"
    trace.write_text(f'''#!/usr/bin/env python3
import json, os, subprocess, sys
result = subprocess.run([{cli!r}] + sys.argv[1:], capture_output=True)
try: report = json.loads(result.stdout)
except (ValueError, UnicodeDecodeError): report = None
entry = json.dumps({{"args":sys.argv[1:],"returncode":result.returncode,"report":report}}, ensure_ascii=False).encode() + b"\\n"
fd = os.open({str(trace_path)!r}, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
os.write(fd, entry); os.close(fd)
sys.stdout.buffer.write(result.stdout); sys.stderr.buffer.write(result.stderr)
sys.exit(result.returncode)
''')
    trace.chmod(0o755)
    transcripts = []
    receipts = []
    def session(inputs, executable=None):
        executable = str(trace) if executable is None else executable
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", executable], cwd=project,
                                env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
        transcripts.append({"stdin":inputs,"stdout":result.stdout,"stderr":result.stderr,"cli_kind":Path(executable).name})
        return result.stdout
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        receipts.append({"args":list(args),"returncode":result.returncode,"report":report})
        return result, report
    initial = source.read_bytes()
    connected = session("\nn\nfixture:model\nfixture:batch\ny\nq\n")
    assert "Project setup preview" in connected, connected
    assert source.read_bytes() == initial
    result, plan = run("config", "set-batch", "--batch", "fixture:batch", "--batch-param", "scale=2")
    assert result.returncode == 0, plan
    result, report = run("config", "set-batch", "--batch", "fixture:batch", "--batch-param", "scale=2", "--apply", "--if-current", plan["data"]["from_revision"])
    assert result.returncode == 0, report
    failed = session("\n2\n\n\nq\n")
    assert "FORWARD_FAILED" in failed and "model.head" in failed and "mat1 and mat2 shapes" in failed, failed
    result, modules = run("layer", "list", "--model", "fixture:model")
    assert result.returncode == 0, modules
    index = next(i for i, item in enumerate(modules["data"]["modules"], 1) if item["path"] == "model.head")
    race = project.parent / "race-cli.py"
    race.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if "--if-source-plan" in sys.argv:
    path = Path("fixture.py")
    path.write_text(path.read_text() + "\\n# concurrent change\\n")
os.execv({str(trace)!r}, [{str(trace)!r}] + sys.argv[1:])
''')
    race.chmod(0o755)
    edit = f'\nh\nset\n{index}\n{{"in_features":48}}\napply\nq\n'
    conflicted = session(edit, str(race))
    assert "SOURCE_PLAN_CONFLICT" in conflicted, conflicted
    assert "in_features=32" in source.read_text() and "# concurrent change" in source.read_text()
    legacy = project.parent / "legacy-cli.py"
    legacy.write_text(f'''#!/usr/bin/env python3
import json, os, sys
if "--if-source-plan" in sys.argv:
    print(json.dumps({{"protocol_version":1,"operation":"edit","status":"error","data":{{}},"diagnostics":[{{"code":"LEGACY_FLAG_UNSUPPORTED","message":"simulated older CLI cannot guard source plans"}}]}}))
    sys.exit(2)
os.execv({str(trace)!r}, [{str(trace)!r}] + sys.argv[1:])
''')
    legacy.chmod(0o755)
    baseline = source.read_bytes()
    rejected = session(edit, str(legacy))
    assert "LEGACY_FLAG_UNSUPPORTED" in rejected and source.read_bytes() == baseline, rejected
    repaired = session(edit)
    assert "Post-save verification" in repaired and '"status":"passed"' in repaired, repaired
    assert "in_features=48" in source.read_text() and "# concurrent change" in source.read_text()
    restarted = session("\n2\n\n\nq\n")
    assert "Observed calls for this batch" in restarted and "model.head" in restarted and "FORWARD_FAILED" not in restarted, restarted
    assert (project / "unrelated.txt").read_text() == "user-owned content\n"
    config = json.loads((project / ".model-maven/project.json").read_text())
    assert config["batch_params"] == {"scale": 2}, config
    evidence_path = os.environ.get("MODEL_MAVEN_JOURNEY_B_EVIDENCE")
    if evidence_path:
        evidence = Path(evidence_path).resolve()
        evidence.mkdir(parents=True, exist_ok=False)
        (evidence / "tui-sessions.json").write_text(json.dumps(transcripts, ensure_ascii=False, indent=2) + "\n")
        (evidence / "cli-receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
        for artifact in (trace_path, trace, race, legacy, source, project / "unrelated.txt", project / ".model-maven/project.json", project / ".model-maven/project.history.json"):
            if artifact.exists(): shutil.copy2(artifact, evidence / artifact.name)
        print(f"Evidence preserved: {evidence}")
    print("Journey B: connect, fail, reject stale and simulated legacy commits, repair source, restart and recheck")
