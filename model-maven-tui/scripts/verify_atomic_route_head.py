#!/usr/bin/env python3
"""Switch a route and its dependent head together through the real TUI."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-atomic-route-head-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    (project / "components.py").write_text("from torch import nn\ndef head6(): return nn.Linear(6, 3)\ndef head8(): return nn.Linear(8, 3)\n")
    def run(*args, success=True):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        assert (proc.returncode == 0) == success, report
        return report
    args = ("scaffold", "--input", "a=4", "--input", "b=2", "--output-dim", "3", "--inject", "head=components:head6")
    plan = run(*args)
    run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    config = project / ".model-maven/project.json"
    original = config.read_bytes()
    run("assembly", "route", "b_encoder", "a", "--dry-run", success=False)
    run("assembly", "set", "head", "--factory", "components:head8", "--dry-run", success=False)
    invalid = run("assembly", "patch", "--route", "b_encoder=a", "--route", "b_encoder=b", success=False)
    assert invalid["diagnostics"][0]["code"] == "SCAFFOLD_ROUTE_INVALID", invalid
    assert config.read_bytes() == original
    def session(confirm):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
            input=f"\na\npatch\nhead=components:head8;@route.b_encoder=a;@output.a_encoder=native\n\n{{}}\ny\n{confirm}\nq\n",
            capture_output=True, text=True, timeout=180, check=True).stdout
    cancelled = session("n")
    assert "Input route overrides" in cancelled and config.read_bytes() == original, cancelled
    saved = session("y")
    assert "Batch assembly preview" in saved and "Assembly saved" in saved, saved
    actual = json.loads(config.read_text())
    assert actual["model_injections"]["head"] == "components:head8", actual
    assert actual["model_params"]["input_routes"]["b_encoder"] == "a", actual
    assert actual["model_params"]["encoder_output_modes"]["a_encoder"] == "native", actual
    run("model", "probe")
    run("try", "--step", "model_starter:step", "--steps", "1")
    generated = run("assembly", "generate")
    assert generated["data"]["generated_validation"]["forward"] == "passed", generated
    restored = subprocess.run(["python3", "-c", "from generated.model_assembly import build_model; import model_starter; m=build_model(); assert m.input_routes['b_encoder']=='a'; assert m(**model_starter.batch()['kwargs']).shape==(2,3)"], cwd=project, env=env, capture_output=True, text=True, timeout=60)
    assert restored.returncode == 0, (restored.stdout, restored.stderr)
    print("TUI atomically switches a dependent input route and head; cancellation, real forward, trial and reconstruction pass")
