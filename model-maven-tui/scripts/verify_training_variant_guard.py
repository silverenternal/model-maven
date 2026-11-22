#!/usr/bin/env python3
"""Confirm variant changes cannot silently alter a TUI training plan."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-training-variant-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), REAL_MAVEN_CLI=cli, COLUMNS="160")
    def run(*args, ok=True):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        assert (result.returncode == 0) == ok, report
        return report
    args = ("scaffold", "--input", "x=4", "--output-dim", "2")
    preview = run(*args)
    run(*args, "--apply", "--if-plan", preview["data"]["plan_revision"])
    (project / "sentinel.py").write_text("from pathlib import Path\nimport model_starter\ndef step(model, batch, context):\n    Path('step-executed').write_text('yes')\n    return model_starter.step(model, batch, context)\nstep.MODEL_MAVEN_STATELESS = True\n")
    run("layer", "set", "model.head", "--param", "bias=false", "--require-forward")
    variant = project / ".model-maven/variants/working.json"
    original = variant.read_bytes()
    shown = run("variant", "show", "working")
    token = shown["data"]["variant_base"]["token"]
    assert len(token) == 64
    wrapper = project / "race-cli"
    wrapper.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if '--if-variant-base' in args:
    path = Path('.model-maven/variants/working.json')
    value = json.loads(path.read_text()); value['concurrent_marker'] = 'changed'
    path.write_text(json.dumps(value))
real = os.environ['REAL_MAVEN_CLI']
os.execv(real, [real] + args)
''')
    wrapper.chmod(0o755)
    inputs = "\ns\n4\n1\nb\nr\ntry-wait\n\nsentinel:step\n\n\n\nn\ny\nq\n"
    def session(executable):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(executable)], cwd=project,
                              env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True).stdout
    raced = session(wrapper)
    assert "VARIANT_BASE_CONFLICT" in raced, raced
    assert not (project / "step-executed").exists()
    assert json.loads(variant.read_text())["concurrent_marker"] == "changed"
    variant.write_bytes(original)
    successful = session(cli)
    assert "Redacted variant snapshot" in successful and "Run started or completed" in successful, successful
    assert (project / "step-executed").read_text() == "yes"
    print("Changed variant rejected before training; matching variant starts and runs the user step")
    (project / "slow.py").write_text("from pathlib import Path\nimport time\nimport model_starter\ndef step(model, batch, context):\n    assert model.head.bias is None\n    Path('slow-started').write_text('yes')\n    time.sleep(8)\n    return model_starter.step(model, batch, context)\nstep.MODEL_MAVEN_STATELESS = True\n")
    receipt = run("train", "--variant", "working", "--if-variant-base", token,
                  "--step", "slow:step", "--steps", "1", "--detach")
    run_id = receipt["data"]["run_id"]
    deadline = time.monotonic() + 30
    while not (project / "slow-started").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (project / "slow-started").exists()
    run("layer", "set", "model.head", "--param", "bias=true", "--require-forward", "--variant", "working")
    while time.monotonic() < deadline:
        status = run("runs", "attach", run_id, "--follow=false")["data"]["record"]["status"]
        if status not in ("running", "stopping"): break
        time.sleep(0.2)
    assert status == "completed", status
    assert run("variant", "show", "working")["data"]["variant_base"]["token"] != token
    print("Detached training uses the confirmed variant snapshot while later variant edits proceed")
