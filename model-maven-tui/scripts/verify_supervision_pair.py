#!/usr/bin/env python3
"""Mark a real dual-input fusion output and preview a rejecting rule via TUI."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-supervision-pair-") as directory:
    project = Path(directory)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    inputs = ("\nm\nimage=3,4,4;tabular=5\n3\ny\n2\n\n\n"
              "t\n2\noutputs.missing\n"
              "t\n2\n\nfused\nFused features\ny\nrange\nrequire\n0\n0.5\n\ny\n"
              "r\ntry-wait\n\n\n\n\n\ny\ny\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli],
                            cwd=project, env=env, input=inputs, capture_output=True,
                            text=True, timeout=180, check=True)
    output = result.stdout
    assert "Port was not observed; nothing saved." in output, output
    assert "Mark saved:" in output, output
    assert "Rule preview" in output and "would_reject" in output, output
    assert "Objective applied: false" in output and "Sample localization: insufficient" in output, output
    assert "Training plan" in output and "Rule:" in output and "STEP_FAILED" in output, output
    state = json.loads((project / ".model-maven/semantic-marks.json").read_text())
    mark = state["marks"][0]
    assert mark["module_path"] == "model.fusion" and mark["port"] == "outputs", mark
    assert mark["role"] == "fused" and mark["rules"][0]["mode"] == "require", mark
    attempt = subprocess.run([cli, "--json", "--no-interactive", "try", "--step", "model_starter:step",
                              "--steps", "1", "--apply-supervision"], cwd=project, env=env,
                             capture_output=True, text=True, timeout=120)
    report = json.loads(attempt.stdout)
    assert attempt.returncode != 0 and report["data"]["cause_code"] == "SUPERVISION_REQUIRE_FAILED", report
    assert report["data"]["completed_steps"] == 0, report
    def manage(inputs):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input=inputs, capture_output=True, text=True, timeout=60, check=True)
    cancelled = manage("\nd\n1\nno\nq\n")
    assert "Saved supervision rules" in cancelled.stdout, cancelled.stdout
    assert len(json.loads((project / ".model-maven/semantic-marks.json").read_text())["marks"][0]["rules"]) == 1
    wrapper = project / "race-cli"
    wrapper.write_text("""#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
real = os.environ['REAL_MAVEN_CLI']
if '--expected-rule' in args:
    expected = json.loads(args[args.index('--expected-rule') + 1])
    common = [real, '--json', '--no-interactive', 'model', 'supervise', 'rule']
    subprocess.run(common + ['remove', '--id', expected['id']], check=True, capture_output=True)
    subprocess.run(common + ['add', '--mark', expected['mark_id'], '--id', expected['id'], '--kind', 'range', '--mode', 'observe', '--min', '10'], check=True, capture_output=True)
os.execv(real, [real] + args)
""")
    wrapper.chmod(0o755)
    race_env = dict(env, REAL_MAVEN_CLI=cli)
    raced = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                           env=race_env, input="\nd\n1\nremove\nq\n", capture_output=True, text=True, timeout=90, check=True)
    assert "SUPERVISION_RULE_CHANGED" in raced.stdout, raced.stdout
    replacement = json.loads((project / ".model-maven/semantic-marks.json").read_text())["marks"][0]["rules"][0]
    assert replacement["min"] == 10 and replacement["mode"] == "observe", replacement
    print("Concurrent replacement after confirmation is retained by conditional removal")
    (project / "sentinel_step.py").write_text("from pathlib import Path\nimport model_starter\ndef step(model, batch, context):\n    Path('step-executed').write_text('executed')\n    return model_starter.step(model, batch, context)\nstep.MODEL_MAVEN_STATELESS = True\n")
    training_race = project / "training-race-cli"
    training_race.write_text("""#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
real = os.environ['REAL_MAVEN_CLI']
if '--if-supervision-revision' in args:
    mark = json.load(open('.model-maven/semantic-marks.json'))['marks'][0]
    rule = mark['rules'][0]
    common = [real, '--json', '--no-interactive', 'model', 'supervise', 'rule']
    subprocess.run(common + ['remove', '--id', rule['id']], check=True, capture_output=True)
    subprocess.run(common + ['add', '--mark', mark['id'], '--id', rule['id'], '--kind', 'range', '--mode', 'observe', '--min', '11'], check=True, capture_output=True)
os.execv(real, [real] + args)
""")
    training_race.chmod(0o755)
    raced_training = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(training_race)], cwd=project,
                                   env=race_env, input="\nr\ntry-wait\n\nsentinel_step:step\n\n\n\ny\ny\nq\n",
                                   capture_output=True, text=True, timeout=120, check=True)
    assert "SUPERVISION_STATE_CONFLICT" in raced_training.stdout, raced_training.stdout
    assert not (project / "step-executed").exists(), raced_training.stdout
    print("Changed rules after training confirmation reject the run before the user step executes")
    removed = manage("\nd\n1\nremove\nq\n")
    assert len(json.loads((project / ".model-maven/semantic-marks.json").read_text())["marks"][0]["rules"]) == 0, removed.stdout
    print("TUI lists saved rules, preserves cancelled removal, and removes only after confirmation")
    print("TUI marks fusion and previews violation; explicitly enabled require rejects the training step before update")

with tempfile.TemporaryDirectory(prefix="maven-supervision-penalty-") as directory:
    project = Path(directory)
    (project / "components.py").write_text('''import json, os
from pathlib import Path
from torch import nn
def encoder():
    layer = nn.Conv2d(3, 3, 1)
    if os.environ.get("GRADIENT_CAPTURE"):
        path = Path(os.environ["GRADIENT_CAPTURE"])
        path.with_suffix(".weights.json").write_text(json.dumps(layer.weight.detach().reshape(-1).tolist()))
        def capture_input(module, inputs):
            path.with_suffix(".input.json").write_text(json.dumps(inputs[0].detach().reshape(-1).tolist()))
        layer.register_forward_pre_hook(capture_input)
        def capture(gradient):
            path.write_text(json.dumps(gradient.detach().reshape(-1).tolist()))
        layer.weight.register_hook(capture)
    return layer
''')
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    inputs = ("\nm\nimage=3,4,4;tabular=5\n3\ny\n"
              "a\n\nimage_encoder\ncomponents:encoder\ny\n2\n\n\n"
              "t\n2\n\nfused\nTrainable fusion output\ny\nrange\npenalty\n0\n0.5\n\n0.1\ny\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli],
                            cwd=project, env=env, input=inputs, capture_output=True,
                            text=True, timeout=180, check=True)
    assert "penalty_preview_only" in result.stdout and "Objective applied: false" in result.stdout, result.stdout
    base_gradient = project / "base-gradient.json"
    penalty_gradient = project / "penalty-gradient.json"
    command = [cli, "--json", "--no-interactive", "try", "--step", "model_starter:step", "--steps", "1"]
    baseline = subprocess.run(command, cwd=project, env=dict(env, GRADIENT_CAPTURE=str(base_gradient)),
                              capture_output=True, text=True, timeout=120)
    assert baseline.returncode == 0, baseline.stdout
    attempt = subprocess.run(command + ["--apply-supervision"], cwd=project,
                             env=dict(env, GRADIENT_CAPTURE=str(penalty_gradient)),
                             capture_output=True, text=True, timeout=120)
    report = json.loads(attempt.stdout)
    assert attempt.returncode == 0 and report["data"]["objective_applied"] is True, report
    state = json.loads((project / ".model-maven/semantic-marks.json").read_text())
    rule_id = state["marks"][0]["rules"][0]["id"]
    assert report["data"]["mean_losses"].get("supervision/" + rule_id, 0) > 0, report
    assert report["data"]["completed_steps"] == 1, report
    base = json.loads(base_gradient.read_text())
    penalized = json.loads(penalty_gradient.read_text())
    for suffix in (".weights.json", ".input.json"):
        assert json.loads(base_gradient.with_suffix(suffix).read_text()) == json.loads(penalty_gradient.with_suffix(suffix).read_text())
    assert len(base) == len(penalized) and any(abs(a - b) > 1e-7 for a, b in zip(base, penalized)), (base, penalized)
    print("Identical trial inputs with supervision disabled/enabled produce different upstream encoder gradients")
    trained = subprocess.run([cli, "--json", "--no-interactive", "train", "--step", "model_starter:step",
                              "--steps", "1", "--checkpoint-every", "1", "--apply-supervision"],
                             cwd=project, env=env, capture_output=True, text=True, timeout=120)
    trained_report = json.loads(trained.stdout)
    assert trained.returncode == 0, trained_report
    with tempfile.TemporaryDirectory(prefix="maven-supervised-export-parent-") as destination:
        exported_path = Path(destination) / "export"
        exported = subprocess.run([cli, "--json", "--no-interactive", "export", str(exported_path), "--run", trained_report["data"]["run_id"]],
                                  cwd=project, env=env, capture_output=True, text=True, timeout=120)
        assert exported.returncode == 0, exported.stdout
        script = "import restore_model as r\ntry: r.resume_managed_steps(1)\nexcept ValueError as e: assert '不支持监督规则' in str(e)\nelse: raise AssertionError('supervised run unexpectedly resumed')\nprint('SUPERVISION_REJECTED')"
        recovered = subprocess.run(["python3", "-c", script], cwd=exported_path, env=env,
                                   capture_output=True, text=True, timeout=120)
        assert recovered.returncode == 0 and "SUPERVISION_REJECTED" in recovered.stdout, (recovered.stdout, recovered.stderr)
        print("Exported continuation rejects supervised checkpoints rather than silently dropping rules")
    print("TUI previews a differentiable fusion penalty; explicitly enabled training applies it")

for kind, details, expected in [
    ("batch-alignment", "batch.kwargs.image\n0\n", {"target_path": "batch.kwargs.image", "axis": 0}),
    ("custom-validator", "0\nconstraints:bounded\n{\"limit\":10}\n", {"axis": 0}),
    ("custom-penalty", "constraints:energy\n{\"target\":0}\n0.25\n", {"weight": 0.25}),
]:
    with tempfile.TemporaryDirectory(prefix="maven-supervision-kind-") as directory:
        project = Path(directory)
        (project / "constraints.py").write_text(
            "def bounded(value, batch, params): return {'passed': True, 'summary': {'limit': params['limit']}}\n"
            "def energy(value, batch, params): return (value - params['target']).square().mean()\n")
        env = os.environ.copy()
        env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
        mode = "penalty" if kind == "custom-penalty" else "observe"
        inputs = ("\nm\nimage=3,4,4;tabular=5\n3\ny\n2\n\n\n"
                  f"t\n2\n\nfused\nFused features\ny\n{kind}\n{mode}\n{details}y\nq\n")
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli],
                                cwd=project, env=env, input=inputs, capture_output=True,
                                text=True, timeout=180, check=True)
        state = json.loads((project / ".model-maven/semantic-marks.json").read_text())
        rule = state["marks"][0]["rules"][0]
        assert rule["kind"] == kind and all(rule[k] == v for k, v in expected.items()), rule
        assert "Rule preview" in result.stdout and f"{kind} · {mode} ·" in result.stdout, result.stdout
        if kind.startswith("custom-"):
            assert "not sandboxed" in result.stdout, result.stdout
            assert rule["plugin"]["entrypoint"].startswith("constraints:"), rule
            assert rule["plugin"]["params"] == ({"target": 0} if kind == "custom-penalty" else {"limit": 10}), rule
        print(f"TUI creates and previews {kind} with explicit parameters")
