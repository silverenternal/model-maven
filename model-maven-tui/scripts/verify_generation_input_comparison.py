#!/usr/bin/env python3
"""Expose whether two real code-generation forwards used the same input."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-generation-input-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    def run(*args):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                              env=env, text=True, capture_output=True, timeout=120, check=True)
        return json.loads(proc.stdout)
    args = ("scaffold", "--input", "x=4", "--output-dim", "2")
    plan = run(*args)
    run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    with (project / "model_starter.py").open("a") as stream:
        stream.write('''\nfrom pathlib import Path
_original_batch = batch
def batch(batch_size=2):
    path = Path("input-value.txt")
    value = float(path.read_text())
    if Path("dynamic").exists(): path.write_text(str(value + 1))
    result = _original_batch(batch_size=batch_size)
    result["kwargs"]["x"] = torch.ones(batch_size, 4) * value
    return result
''')
    (project / "input-value.txt").write_text("1")
    stable = run("assembly", "generate", "--output", "generated/stable.py")
    assert stable["data"]["input_comparison"]["status"] == "same", stable
    (project / "dynamic").touch()
    output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                            env=env, input="\na\ngenerate\n\ny\nq\n", text=True,
                            capture_output=True, timeout=180, check=True).stdout
    assert "Validation inputs: different" in output, output
    assert (project / "generated/model_assembly.py").is_file() and float((project / "input-value.txt").read_text()) == 3, output
    cfg_path = project / ".model-maven/project.json"
    cfg = json.loads(cfg_path.read_text())
    for key in ("batch", "batch_binding", "batch_params"): cfg.pop(key, None)
    cfg_path.write_text(json.dumps(cfg))
    no_batch = run("assembly", "generate", "--output", "generated/no_batch.py")
    assert no_batch["data"]["input_comparison"]["status"] == "not_run", no_batch
    print("Generated-code validation distinguishes same, different, and absent inputs using real forwards")
