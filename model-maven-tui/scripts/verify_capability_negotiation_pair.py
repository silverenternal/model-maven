#!/usr/bin/env python3
"""Verify TUI feature gating using real CLI with altered discovery receipts."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-capabilities-") as directory:
    top = Path(directory)
    wrapper = top / "wrapped cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
with Path("calls.jsonl").open("a") as stream: stream.write(json.dumps(sys.argv[1:]) + "\\n")
mode = os.environ.get("CAP_MODE", "")
if mode == "wrong_success_operation" and "model" in sys.argv and ("inspect" in sys.argv or "probe" in sys.argv):
    print(json.dumps({{"protocol_version":1,"operation":"doctor","status":"success","data":{{}}}})); sys.exit(0)
if mode == "legacy_no_config_revision" and "assembly" in sys.argv and "show" in sys.argv and "--config-only" in sys.argv:
    output = subprocess.run([{cli!r}, *sys.argv[1:]], capture_output=True, text=True)
    report = json.loads(output.stdout)
    report["data"].pop("config_revision", None)
    print(json.dumps(report)); sys.exit(output.returncode)
if "--describe-interface" in sys.argv:
    if mode == "oversized":
        sys.stdout.buffer.write(b"x" * (32 * 1024 * 1024 + 1)); sys.exit(0)
    if mode in ("legacy", "legacy_no_config_revision"):
        print(json.dumps({{"protocol_version":1,"operation":"cli","status":"error","diagnostics":[{{"code":"CLI_ARGUMENT_INVALID","message":"unsupported option"}}]}})); sys.exit(2)
    output = subprocess.run([{cli!r}, *sys.argv[1:]], capture_output=True, text=True)
    report = json.loads(output.stdout)
    caps = report["data"]["capabilities"]
    if mode.startswith("missing:"): caps.pop(mode.removeprefix("missing:"), None)
    if mode.startswith("bad:"): caps[mode.removeprefix("bad:")] = "yes"
    if mode == "missing_source": caps.pop("edit.conditional_source_config", None)
    if mode == "bad_source": caps["edit.conditional_source_config"] = "yes"
    if mode == "missing_objectives": caps.pop("project.scaffold_objectives", None)
    if mode == "missing_weights": caps.pop("project.scaffold_objective_weights", None)
    if mode == "missing_module": caps.pop("project.scaffold_custom_module", None)
    if mode == "missing_external_batch": caps.pop("project.scaffold_external_batch", None)
    if mode == "missing_batch_params": caps.pop("project.scaffold_external_batch_params", None)
    if mode == "missing_initial_bindings": caps.pop("project.scaffold_initial_bindings", None)
    if mode == "missing_codegen_verify": caps.pop("assembly.generated_source_verify", None)
    if mode == "missing_native_encoder": caps.pop("project.scaffold_native_encoder_output", None)
    if mode == "missing_native_edit": caps.pop("project.scaffold_encoder_output_edit", None)
    if mode == "missing_native_edit_evidence": caps.pop("project.scaffold_encoder_output_evidence", None)
    if mode == "missing_atomic_encoder": caps.pop("assembly.atomic_patch_encoder_output", None)
    if mode == "missing_atomic_route": caps.pop("assembly.atomic_patch_route", None)
    if mode == "missing_route_evidence": caps.pop("project.scaffold_route_evidence", None)
    if mode == "missing_setting_snapshot": caps.pop("project.scaffold_setting_snapshot", None)
    if mode == "missing_typed": caps.pop("data.csv_batch_typed", None)
    print(json.dumps(report)); sys.exit(output.returncode)
os.execv({cli!r}, [{cli!r}, *sys.argv[1:]])
''')
    wrapper.chmod(0o755)
    env = dict(os.environ, COLUMNS="180")
    def new_project(name):
        path = top / name
        path.mkdir()
        return path
    def run(project, *args):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        assert proc.returncode == 0 and report["status"] == "success", report
        return report
    def session(project, mode, inputs):
        local = dict(env, CAP_MODE=mode, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
        proc = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project, env=local,
                              input=inputs, capture_output=True, text=True, timeout=120, check=True)
        return proc.stdout
    project = new_project("source")
    source = project / "fixture.py"
    source.write_text("import torch\nfrom torch import nn\ndef model(): return nn.Sequential(nn.Linear(4, 2))\ndef batch(): return torch.ones(2, 4)\n")
    run(project, "init", "--model", "fixture:model", "--batch", "fixture:batch")
    original = source.read_bytes()
    output = session(project, "wrong_success_operation", "\n1\n\n2\n\n\nq\n")
    assert "CLI protocol error: expected inspect report, received doctor" in output, output
    assert "CLI protocol error: expected probe report, received doctor" in output, output
    assert source.read_bytes() == original, output
    for mode, expected in [("missing_source", "edit.conditional_source_config"), ("bad_source", "Could not verify CLI capabilities"), ("legacy", "does not provide a capability catalog"), ("oversized", "exceeds 32 MiB")]:
        output = session(project, mode, "\nh\nset\nq\n")
        assert expected in output and source.read_bytes() == original, (mode, output)
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("layer" in args and "set" in args for args in calls), calls
    output = session(project, "missing_native_edit", "\na\nencoder-output\nq\n")
    assert "project.scaffold_encoder_output_edit" in output and source.read_bytes() == original, output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("encoder-output" in args for args in calls), calls
    output = session(project, "missing_native_edit_evidence", "\na\nencoder-output\nq\n")
    assert "project.scaffold_encoder_output_evidence" in output and source.read_bytes() == original, output
    output = session(project, "missing_atomic_encoder", "\na\npatch\n@output.text_encoder=native\n\n{}\nq\n")
    assert "assembly.atomic_patch_encoder_output" in output and source.read_bytes() == original, output
    output = session(project, "missing_atomic_route", "\na\npatch\n@route.b_encoder=a\n\n{}\nq\n")
    assert "assembly.atomic_patch_route" in output and source.read_bytes() == original, output
    output = session(project, "missing_route_evidence", "\na\nroute\nq\n")
    assert "project.scaffold_route_evidence" in output and source.read_bytes() == original, output
    output = session(project, "missing_setting_snapshot", "\na\nroute+snapshot\nq\n")
    assert "project.scaffold_setting_snapshot" in output and source.read_bytes() == original, output
    project = new_project("missing-codegen-verify")
    (project / "fixture.py").write_text("import torch\nfrom torch import nn\ndef model(): return nn.Linear(4, 2)\ndef batch(): return torch.ones(2, 4)\n")
    run(project, "init", "--model", "fixture:model", "--batch", "fixture:batch")
    output = session(project, "missing_codegen_verify", "\na\ngenerate\nq\n")
    assert "assembly.generated_source_verify" in output and not (project / "generated/model_assembly.py").exists(), output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("assembly" in args and "generate" in args for args in calls), calls
    project = new_project("legacy-history")
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
    run(project, "init", "--model", "fixture:model", "--batch", "fixture:batch")
    original = source.read_bytes()
    run(project, "layer", "set", "model.head", "--target", "source", "--param", "bias=false", "--require-forward")
    assert source.read_bytes() != original
    output = session(project, "legacy_no_config_revision", "\nh\nundo\napply\nq\n")
    assert "separate forward recheck" in output and source.read_bytes() == original, output
    project = new_project("objectives")
    output = session(project, "missing_objectives", "\nm\nx=4\nce:3\nq\n")
    assert "project.scaffold_objectives" in output and not (project / "model_starter.py").exists(), output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("scaffold" in args for args in calls), calls
    project = new_project("missing-native-encoder")
    output = session(project, "missing_native_encoder", "\nm\nx=?5,4;@native=x_encoder\n1\nq\n")
    assert "project.scaffold_native_encoder_output" in output and not (project / "model_starter.py").exists(), output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("scaffold" in args for args in calls), calls
    project = new_project("missing-initial-bindings")
    output = session(project, "missing_initial_bindings", "\nm\nx=4;@seam.x_encoder=layers:make\n1\nq\n")
    assert "project.scaffold_initial_bindings" in output and not (project / "model_starter.py").exists(), output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("scaffold" in args for args in calls), calls
    project = new_project("missing-batch-params")
    output = session(project, "missing_batch_params", "\nm\nx=4;@batch=user_data:batch;@batchparam.size=2\n1\nq\n")
    assert "project.scaffold_external_batch_params" in output and not (project / "model_starter.py").exists(), output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("scaffold" in args for args in calls), calls
    project = new_project("missing-external-batch")
    output = session(project, "missing_external_batch", "\nm\nx=4;@batch=user_data:batch\n1\nq\n")
    assert "project.scaffold_external_batch" in output and not (project / "model_starter.py").exists(), output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("scaffold" in args for args in calls), calls
    project = new_project("missing-module")
    output = session(project, "missing_module", "\nm\nx=4;@module=models.custom\n1\nq\n")
    assert "project.scaffold_custom_module" in output and not (project / "models/custom.py").exists(), output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("scaffold" in args for args in calls), calls
    project = new_project("missing-weights")
    output = session(project, "missing_weights", "\nm\nx=4\nscore=1:mse@0.5\nq\n")
    assert "project.scaffold_objective_weights" in output and not (project / "model_starter.py").exists(), output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("scaffold" in args for args in calls), calls
    project = new_project("explicit-mse")
    output = session(project, "missing_objectives", "\nm\nx=4\nscore=1:mse\nq\n")
    assert "project.scaffold_objectives" in output and not (project / "model_starter.py").exists(), output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("scaffold" in args for args in calls), calls
    project = new_project("legacy-basic")
    output = session(project, "legacy", "\nm\nq\n")
    assert "does not provide a capability catalog" in output and not (project / "model_starter.py").exists(), output
    for capability, inputs in [
        ("project.classification_create", "\nb\nq\n"),
        ("project.custom_scaffold", "\nm\nq\n"),
        ("project.scaffold_input_routes", "\nm\nx=4;y=2;@x_encoder=y\n1\nq\n"),
    ]:
        project = new_project("missing-" + capability)
        output = session(project, "missing:" + capability, inputs)
        assert capability in output and not (project / "model_starter.py").exists(), output
        calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
        assert not any("scaffold" in args or "create" in args for args in calls), calls
    project = new_project("missing-base-edits")
    (project / "fixture.py").write_text("import torch\nfrom torch import nn\ndef model(): return nn.Linear(4, 2)\ndef batch(): return torch.ones(2, 4)\n")
    run(project, "init", "--model", "fixture:model", "--batch", "fixture:batch")
    config_path = project / ".model-maven/project.json"
    original = config_path.read_bytes()
    for capability, inputs in [
        ("assembly.conditional_config_edit", "\na\nadd\nq\n"),
        ("project.conditional_batch_edit", "\na\nbatch\nq\n"),
        ("augmentation.conditional_config_edit", "\nx\naugment\n\nadd\nb\nq\n"),
    ]:
        output = session(project, "missing:" + capability, inputs)
        assert capability in output and config_path.read_bytes() == original, output
    output = session(project, "bad:assembly.conditional_config_edit", "\na\nadd\nq\n")
    assert "Could not verify CLI capabilities" in output and config_path.read_bytes() == original, output
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any(("assembly" in args and "add" in args) or ("set-batch" in args) or
                   ("augment" in args and "add" in args) or ("csv-template" in args) for args in calls), calls
    project = new_project("typed")
    plan = run(project, "scaffold", "--input", "x=2", "--output-dim", "1")
    run(project, "scaffold", "--input", "x=2", "--output-dim", "1", "--apply", "--if-plan", plan["data"]["plan_revision"])
    original = (project / ".model-maven/project.json").read_bytes()
    output = session(project, "missing_typed", '\na\nbatch\n:csv\ninput.csv\n{"x":{"columns":["a","b"],"dtype":"int64"}}\n["y"]\n2\nq\n')
    assert "data.csv_batch_typed" in output and not (project / "model_csv.py").exists(), output
    assert (project / ".model-maven/project.json").read_bytes() == original
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert not any("csv-template" in args for args in calls), calls
    print("TUI gates project creation and guarded edits on advertised capabilities; source history remains available")
