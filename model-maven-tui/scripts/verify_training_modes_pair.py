#!/usr/bin/env python3
"""Exercise custom manual-step and external-loop through the real workbench."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-training-modes-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"))
    def command(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=90)
        return result, json.loads(result.stdout)
    scaffold = ("scaffold", "--input", "image=3,4,4", "--input", "tabular=5", "--output-dim", "3")
    result, preview = command(*scaffold)
    assert result.returncode == 0, preview
    result, report = command(*scaffold, "--apply", "--if-plan", preview["data"]["plan_revision"])
    assert result.returncode == 0, report
    (project / "user_training.py").write_text('''import model_starter

def manual(model, batch, context, scale=1):
    assert scale == 2
    optimizer = context.optimizer("all", model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss = (model(**batch["kwargs"]) - batch["loss_inputs"]["target"]).square().mean() * scale
    context.backward(loss)
    optimizer.step()
    return {"loss": loss.detach()}
manual.MODEL_MAVEN_STATELESS = True

def loop(model, context, scale=1):
    assert scale == 3
    import torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    while context.remaining_steps:
        batch = model_starter.batch()
        optimizer.zero_grad()
        loss = (model(**batch["kwargs"]) - batch["loss_inputs"]["target"]).square().mean() * scale
        loss.backward()
        optimizer.step()
        if not context.report_step(loss.detach(), {"scale": scale}):
            break
loop.MODEL_MAVEN_STATELESS = True
''')
    inputs = ("\nr\ntry-wait\nmanual-step\nuser_training:manual\n{\"scale\":2}\n1\n\nn\ny\n"
              "r\ntrain\nexternal-loop\nuser_training:loop\n{\"scale\":3}\n2\n\nn\ny\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                            env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    output = result.stdout
    assert output.count("Run started or completed") == 2, (output, result.stderr)
    assert "Mode: manual-step" in output and "Mode: external-loop" in output, output
    assert "Batch: managed by user loop" in output and "\"scale\":3" in output, output
    _, listing = command("runs", "list")
    trial = next(item for item in listing["data"]["runs"] if item["operation"] == "training_probe")
    trial_record = json.loads((project / ".model-maven/runs" / trial["id"] / "run.json").read_text())
    assert trial_record["training_mode"] == "manual-step", trial_record
    run = next(item for item in listing["data"]["runs"] if item["operation"] == "train")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        _, detail = command("runs", "attach", run["id"], "--follow=false")
        status = detail["data"]["record"]["status"]
        if status not in ("running", "stopping"):
            break
        time.sleep(0.3)
    assert status == "completed", detail
    record = json.loads((project / ".model-maven/runs" / run["id"] / "run.json").read_text())
    assert record["training_mode"] == "external-loop", record
    assert (project / ".model-maven/runs" / run["id"] / "checkpoints/latest.json").is_file(), record
    with tempfile.TemporaryDirectory(prefix="maven-external-export-parent-") as parent:
        destination = Path(parent) / "export"
        result, exported = command("export", str(destination), "--run", run["id"])
        assert result.returncode == 0, exported
        check = "import restore_model as r\ntry: r.resume_managed_steps(1)\nexcept ValueError as e: assert 'managed-step' in str(e)\nelse: raise AssertionError('external loop unexpectedly accepted')\nprint('EXTERNAL_REJECTED')"
        isolated = subprocess.run(["python3", "-c", check], cwd=destination, env=env,
                                  capture_output=True, text=True, timeout=90)
        assert isolated.returncode == 0 and "EXTERNAL_REJECTED" in isolated.stdout, (isolated.stdout, isolated.stderr)
        print("Exported continuation rejects external-loop checkpoints")
    print("TUI runs a custom manual step and detached user-owned loop with JSON parameters")
