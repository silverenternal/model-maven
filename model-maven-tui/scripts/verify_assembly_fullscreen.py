#!/usr/bin/env python3
"""Create and repair a dual-input assembly through a real full-screen TUI."""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import termios
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-assembly-screen-") as directory:
    project = Path(directory) / "project"
    project.mkdir()
    destination = Path(directory) / "export"
    env = dict(os.environ, TERM="xterm-256color", NO_COLOR="1", MODEL_MAVEN_REDUCE_MOTION="1",
               MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    env.pop("COLUMNS", None)
    env.pop("TMUX", None)
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    child = subprocess.Popen([tui, "--lang", "en", "--cli", cli], cwd=project, env=env,
                             stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    pending, transcript = bytearray(), bytearray()

    def wait(token, timeout=30):
        needle = token.encode()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos = pending.find(needle)
            if pos >= 0:
                del pending[:pos + len(needle)]
                return
            if child.poll() is not None:
                raise AssertionError(f"TUI exited: {pending[-6000:]!r}")
            if select.select([master], [], [], 0.1)[0]:
                chunk = os.read(master, 65536)
                pending.extend(chunk)
                transcript.extend(chunk)
        raise AssertionError(f"missing {token!r}: {pending[-6000:]!r}")

    def send(value):
        os.write(master, value)

    def assembly(slot, factory, succeeds):
        send(b"a")
        wait("Saved bindings")
        send(b"b")
        wait("Action: add, set")
        send(b"\n")
        wait("Factory seam name")
        send(slot.encode() + b"\n")
        wait("Python factory module:callable")
        send(factory.encode() + b"\n")
        if succeeds:
            wait("Assembly preview")
            send(b"b")
            wait("Save assembly? Type y")
            send(b"y\n")
            wait("Assembly saved")
        else:
            wait("FORWARD_FAILED")
        send(b"b")
        wait("Model-Maven Workbench")

    try:
        wait("Open workbench")
        send(b"\n")
        wait("Model-Maven Workbench")
        send(b"m")
        wait("Inputs and options")
        send(b"image=3,4,4;tabular=5\n")
        wait("Output features")
        send(b"3\n")
        wait("Custom model preview")
        send(b"b")
        wait("Create files?")
        send(b"y\n")
        wait("Custom model project created.")
        send(b"b")
        wait("Model-Maven Workbench")
        (project / "components.py").write_text(
            "import torch\nfrom torch import nn\ndef broken(): return nn.Linear(2, 2)\n"
            "def image(): return nn.ReLU()\ndef tabular(): return nn.Identity()\n"
            "class Fusion(nn.Module):\n    def forward(self, values): return torch.cat(values, dim=1)\n"
            "def fusion(): return Fusion()\ndef head(): return nn.Linear(53, 3)\n")
        (project / "evaluation.py").write_text(
            "import model_starter\ndef data(): return model_starter.batch()\n"
            "def evaluate(model, data, context):\n"
            "    output = model(**data['kwargs'])\n    context.report_samples(2)\n"
            "    context.report_prediction('example-0', output[0].tolist())\n"
            "    return {'metrics': {'mean_abs': float(output.abs().mean())}}\n")
        assembly("image_encoder", "components:broken", False)
        assembly("image_encoder", "components:image", True)
        assembly("tabular_encoder", "components:tabular", True)
        assembly("fusion", "components:fusion", True)
        assembly("head", "components:head", True)
        config = json.loads((project / ".model-maven/project.json").read_text())
        assert config["model_injections"]["image_encoder"] == "components:image", config
        assert config["model_injections"]["tabular_encoder"] == "components:tabular", config
        assert config["model_injections"]["fusion"] == "components:fusion", config
        assert config["model_injections"]["head"] == "components:head", config
        send(b"a")
        wait("Saved bindings")
        send(b"b")
        wait("Action: add, set")
        send(b"\x15generate\n")
        wait("New project-relative Python file")
        send(b"\n")
        wait("Generate assembly adapter")
        send(b"b")
        wait("Generate file? Type y")
        send(b"y\n")
        wait("Generated code re-import and forward")
        wait('"forward":"passed"')
        send(b"b")
        wait("Model-Maven Workbench")
        generated = project / "generated/model_assembly.py"
        assert generated.is_file()
        rebuilt = subprocess.run(["python3", "-c", "from generated.model_assembly import build_model; "
                                  "import model_starter; m=build_model(); "
                                  "assert tuple(m(**model_starter.batch()['kwargs']).shape)==(2,3)"],
                                 cwd=project, env=env, capture_output=True, text=True, timeout=90)
        assert rebuilt.returncode == 0, (rebuilt.stdout, rebuilt.stderr)
        send(b"2\n")
        wait("Batch factory")
        send(b"\n")
        wait("Completed")
        result = subprocess.run([cli, "--json", "--no-interactive", "model", "probe"], cwd=project,
                                env=env, capture_output=True, text=True, timeout=90, check=True)
        report = json.loads(result.stdout)
        assert report["status"] == "success", report
        calls = []
        for call in report["data"]["calls"]:
            source = call.get("implementation", {}).get("source_file")
            paths = call.get("paths", [])
            ports = [p for p in call.get("input_output_paths", []) if p.startswith("outputs")]
            if source and not source.startswith("@") and paths and ports:
                calls.append(paths[0])
        fusion_call = calls.index("model.fusion") + 1
        send(b"b")
        wait("Model-Maven Workbench")
        send(b"t")
        wait("Observed project-source calls")
        wait("model.fusion")
        send(b"b")
        wait("Call number")
        send(f"{fusion_call}\n".encode())
        wait("Observed output port")
        send(b"\n")
        wait("Semantic role")
        send(b"fused\n")
        wait("Meaning")
        send(b"Finite fused representation\n")
        wait("Save mark after a new forward? Type y")
        send(b"y\n")
        wait("Rule: finite")
        send(b"finite\n")
        wait("Mode: observe")
        send(b"\x15require\n")
        wait("Sample axis")
        send(b"\n")
        wait("Rule affects training only")
        send(b"y\n")
        wait("Rule preview")
        wait("finite")
        send(b"b")
        wait("Model-Maven Workbench")
        marks = json.loads((project / ".model-maven/semantic-marks.json").read_text())
        assert marks["marks"][0]["module_path"] == "model.fusion", marks
        assert marks["marks"][0]["rules"][0]["kind"] == "finite", marks
        assert marks["marks"][0]["rules"][0]["mode"] == "require", marks
        send(b"r")
        wait("Run type: try")
        send(b"\x15try-wait\n")
        wait("Training mode:")
        send(b"\n")
        wait("Training step module:callable")
        send(b"\n")
        wait("Step/loop parameters")
        send(b"\n")
        wait("Number of steps")
        send(b"\n")
        wait("Device: cpu")
        send(b"\n")
        wait("Apply saved supervision rules?")
        send(b"\x15y\n")
        wait("Training plan")
        wait("Supervision: y")
        send(b"b")
        wait("Start this run? Type y")
        send(b"y\n")
        wait("Run started or completed", timeout=120)
        wait("Status: completed")
        def command(*args):
            result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                    env=env, capture_output=True, text=True, timeout=90, check=True)
            return json.loads(result.stdout)
        listing = command("runs", "list")
        trial = next(item for item in listing["data"]["runs"] if item["operation"] == "training_probe")
        trial_report = command("runs", "show", trial["id"])
        assert trial_report["data"]["inputs"]["apply_supervision"] is True, trial_report
        rules = trial_report["data"]["report"]["data"]["supervision"]["rules"]
        assert any(rule["kind"] == "finite" and rule["mode"] == "require"
                   and rule["steps_observed"] == 1 and rule["element_count"] > 0 for rule in rules), rules
        send(b"b")
        wait("Model-Maven Workbench")
        train_ids, train_reports = [], []
        for _ in range(2):
            previous = {item["id"] for item in command("runs", "list")["data"]["runs"]}
            send(b"r")
            wait("Run type: try")
            send(b"\x15train\n")
            wait("Training mode:")
            send(b"\n")
            wait("Training step module:callable")
            send(b"\n")
            wait("Step/loop parameters")
            send(b"\n")
            wait("Number of steps")
            send(b"\n")
            wait("Device: cpu")
            send(b"\n")
            wait("Apply saved supervision rules?")
            send(b"\x15y\n")
            wait("Training plan")
            wait("Supervision: y")
            send(b"b")
            wait("Start this run? Type y")
            send(b"y\n")
            wait("Run started or completed", timeout=120)
            wait("Run ID:")
            current = command("runs", "list")["data"]["runs"]
            added = [item for item in current if item["id"] not in previous and item["operation"] == "train"]
            assert len(added) == 1, current
            run_id = added[0]["id"]
            train_ids.append(run_id)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                detail = command("runs", "attach", run_id, "--follow=false")
                status = detail["data"]["record"]["status"]
                if status not in ("running", "stopping"):
                    break
                time.sleep(0.25)
            assert status == "completed", detail
            saved = command("runs", "show", run_id)
            assert saved["data"]["inputs"]["apply_supervision"] is True, saved
            rules = saved["data"]["report"]["data"]["supervision"]["rules"]
            assert any(rule["kind"] == "finite" and rule["mode"] == "require"
                       and rule["steps_observed"] == 1 and rule["element_count"] > 0 for rule in rules), rules
            train_reports.append(saved)
            send(b"b")
            wait("Model-Maven Workbench")
        listing = command("runs", "list")["data"]["runs"]
        chosen = next(i for i, item in enumerate(listing, 1) if item["id"] == train_ids[1])
        other = next(i for i, item in enumerate(listing, 1) if item["id"] == train_ids[0])
        send(b"j")
        wait("Training runs")
        send(b"b")
        wait("Run number, refresh")
        send(f"{chosen}\n".encode())
        wait("Run status")
        wait("Supervision measurements")
        send(b"b")
        wait("refresh, watch, stop")
        send(b"\x15compare\n")
        wait("Compare with run number")
        send(f"{other}\n".encode())
        wait("Run comparison")
        wait("data: unknown")
        send(b"b")
        wait("Run status")
        send(b"b")
        wait("refresh, watch, stop")
        send(b"\x15evaluate\n")
        wait("Evaluator module:callable")
        send(b"evaluation:evaluate\n")
        wait("Evaluation data factory module:callable")
        send(b"evaluation:data\n")
        wait("Data split")
        send(b"\n")
        wait("Data version / fingerprint")
        send(b"\n")
        wait("Maximum evaluation samples")
        send(b"\n")
        wait("Prediction previews")
        send(b"\n")
        wait("Evaluator parameters")
        send(b"\n")
        wait("Data factory parameters")
        send(b"\n")
        wait("Evaluation plan")
        send(b"b")
        wait("Run evaluation? Type y")
        send(b"y\n")
        wait("Evaluation result", timeout=120)
        wait("example-0")
        send(b"b")
        wait("Run status")
        send(b"b")
        wait("refresh, watch, stop")
        send(b"\x15export\n")
        wait("New export directory outside the project")
        send(str(destination).encode() + b"\n")
        wait("Export plan")
        send(b"b")
        wait("Export this run? Type y")
        send(b"y\n")
        wait("Export completed", timeout=120)
        assert (destination / "manifest.json").is_file()
        hidden = project.with_name("hidden-project")
        project.rename(hidden)
        try:
            restored = subprocess.run([sys.executable, "restore_model.py"], cwd=destination, env=env,
                                      capture_output=True, text=True, timeout=120)
        finally:
            hidden.rename(project)
        assert restored.returncode == 0, (restored.stdout, restored.stderr)
        evidence = os.environ.get("MODEL_MAVEN_INTEGRATION_ARTIFACTS_DIR")
        if evidence:
            evidence = Path(evidence)
            evidence.mkdir(parents=True, exist_ok=False)
            (evidence / "terminal.ansi").write_bytes(transcript)
            (evidence / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            (evidence / "trial.json").write_text(json.dumps(trial_report, ensure_ascii=False, indent=2) + "\n")
            for index, saved in enumerate(train_reports, 1):
                (evidence / f"train-{index}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
            shutil.copy2(project / ".model-maven/project.json", evidence / "project.json")
            shutil.copy2(project / "model_starter.py", evidence / "model_starter.py")
            shutil.copy2(project / "components.py", evidence / "components.py")
            shutil.copy2(project / "evaluation.py", evidence / "evaluation.py")
            shutil.copy2(generated, evidence / "model_assembly.py")
            shutil.copy2(project / ".model-maven/semantic-marks.json", evidence / "semantic-marks.json")
            shutil.copytree(destination, evidence / "export")
        send(b"b")
        wait("Run status")
        send(b"b")
        wait("refresh, watch, stop")
        send(b"\x1b")
        wait("Training runs")
        send(b"b")
        wait("Run number, refresh")
        send(b"\x1b")
        wait("Model-Maven Workbench")
        send(b"q")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("Full-screen: create, repair, supervise, train twice, compare, evaluate, export and restore")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
