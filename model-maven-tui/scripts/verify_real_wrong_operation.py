#!/usr/bin/env python3
"""Verify unexpected operations and inconsistent recheck evidence are not trusted."""
import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))

for case in ("preview", "commit", "recheck_content", "recheck_reused", "recheck_truncated"):
    with tempfile.TemporaryDirectory(prefix="maven-wrong-operation-") as directory:
        project = Path(directory)
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
        wrapper = project / "cli_wrapper.py"
        wrapper.write_text(f'''#!/usr/bin/env python3
import json, subprocess, sys
result = subprocess.run([{cli!r}] + sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stderr.buffer.write(result.stderr)
args = sys.argv[1:]
is_edit = "layer" in args and "set" in args
is_preview = "--dry-run" in args
if is_edit and (({case!r} == "preview" and is_preview) or ({case!r} == "commit" and not is_preview)):
    report = json.loads(result.stdout)
    report["operation"] = "probe"
    sys.stdout.write(json.dumps(report, ensure_ascii=False))
elif {case!r} in ("recheck_content", "recheck_reused", "recheck_truncated") and "probe" in args and "--if-variant-base" in args:
    report = json.loads(result.stdout)
    if {case!r} == "recheck_content":
        report["data"]["batch_content"]["digest"] = "0" * 64
    elif {case!r} == "recheck_reused":
        report["data"]["evidence_cache"]["status"] = "reused"
    else:
        report["data"]["trace_truncated"] = True
    sys.stdout.write(json.dumps(report, ensure_ascii=False))
else:
    sys.stdout.buffer.write(result.stdout)
sys.exit(result.returncode)
''', encoding="utf-8")
        wrapper.chmod(0o755)
        env = os.environ.copy()
        env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
        action = "y\n0\n" if case != "preview" else "0\n"
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)],
                                cwd=project, env=env,
                                input="\n1\nfixture:model\n2\n\nfixture:batch\n7\n48\n" + action + "q\n",
                                capture_output=True, text=True, timeout=150, check=True)
        output = result.stdout
        variant = (project / ".model-maven/variants/working.json").exists()
        if case == "preview":
            assert "Guarded recheck" not in output, output
            assert "Unexpected CLI operation" in output, output
            assert "Bound input content matches diagnosis" not in output, output
            assert "Commit guarded variant?" not in output, output
            assert not variant
        elif case == "commit":
            assert "Guarded recheck" not in output, output
            assert "Commit guarded variant?" in output, output
            assert "Commit not confirmed" in output, output
            assert variant  # The wrapper changed the response after the real write.
        else:
            assert "Guarded recheck" in output, output
            if case == "recheck_content":
                assert "Recheck input content differs from diagnosis" in output, output
            elif case == "recheck_reused":
                assert "Recheck reused prior evidence" in output, output
            else:
                assert "Recheck call trace was truncated" in output, output
            assert variant
        print(f"{case}: inconsistent response was not treated as verified")
