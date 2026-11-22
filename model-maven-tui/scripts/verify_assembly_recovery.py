#!/usr/bin/env python3
"""Repair an assembly whose saved factory now fails a real forward."""
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-assembly-repair-") as directory:
    project = Path(directory)
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=120)
        return result, json.loads(result.stdout)
    args = ("scaffold", "--input", "image=3,4,4", "--input", "tabular=5", "--output-dim", "3")
    preview, report = run(*args)
    assert preview.returncode == 0, report
    assert run(*args, "--apply", "--if-plan", report["data"]["plan_revision"])[0].returncode == 0
    factory = project / "components.py"
    factory.write_text("from torch import nn\ndef encoder(): return nn.Identity()\ndef good(): return nn.ReLU()\n")
    added, report = run("assembly", "add", "image_encoder", "--factory", "components:encoder")
    assert added.returncode == 0, report
    factory.write_text("from torch import nn\ndef encoder(): return nn.Linear(2, 2)\ndef good(): return nn.ReLU()\n")
    failed, report = run("assembly", "show")
    assert failed.returncode != 0 and report["diagnostics"][0]["code"] == "FORWARD_FAILED", report
    config, report = run("--python", "/definitely/missing/python", "assembly", "show", "--config-only")
    assert config.returncode == 0 and report["data"]["validation"] == {"construction": "not_run", "forward": "not_run"}, report
    assert report["data"]["slots"][0]["slot"] == "image_encoder", report
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
                            input="\na\nset\nimage_encoder\ncomponents:good\ny\n2\n\n\nq\n",
                            capture_output=True, text=True, timeout=180, check=True)
    assert "Configuration only; current model has not been loaded or verified." in result.stdout, result.stdout
    assert "Assembly saved" in result.stdout and "ReLU" in result.stdout, result.stdout
    assert json.loads((project / ".model-maven/project.json").read_text())["model_injections"]["image_encoder"] == "components:good"
    print("Broken forward: configuration-only listing, verified replacement and TUI re-probe passed")

with tempfile.TemporaryDirectory(prefix="maven-assembly-multiple-repair-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"))
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120)
        return result, json.loads(result.stdout)
    args = ("scaffold", "--input", "image=3,4,4", "--input", "tabular=5", "--output-dim", "3")
    result, preview = run(*args)
    assert result.returncode == 0, preview
    assert run(*args, "--apply", "--if-plan", preview["data"]["plan_revision"])[0].returncode == 0
    starter = project / "model_starter.py"
    with starter.open("a") as stream:
        stream.write('''\nfrom pathlib import Path\n_original_batch = batch\ndef batch(batch_size=2):\n    result = _original_batch(batch_size=batch_size)\n    result["kwargs"]["image"] *= float(Path("input-value.txt").read_text())\n    return result\n''')
    (project / "input-value.txt").write_text("1")
    factory = project / "components.py"
    factory.write_text("from torch import nn\ndef first(): return nn.Identity()\ndef second(): return nn.Identity()\ndef good(): return nn.ReLU()\n")
    for slot, entry in (("image_encoder", "first"), ("tabular_encoder", "second")):
        result, report = run("assembly", "add", slot, "--factory", f"components:{entry}")
        assert result.returncode == 0, report
    factory.write_text("from torch import nn\ndef first(): return nn.Linear(2, 2)\ndef second(): return nn.Linear(2, 2)\ndef good(): return nn.ReLU()\ndef scaled(slope=0.1): return nn.LeakyReLU(slope)\n")
    config_path = project / ".model-maven/project.json"
    original = config_path.read_bytes()
    for slot in ("image_encoder", "tabular_encoder"):
        result, report = run("assembly", "set", slot, "--factory", "components:good", "--dry-run")
        assert result.returncode != 0 and report["status"] == "error", report
    inputs = "image_encoder=components:good;tabular_encoder=components:good"
    def session(answer, cli_path=cli, pin="n"):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(cli_path)], cwd=project, env=env,
                              input=f"\na\npatch\n{inputs}\n\n{{}}\n{pin}\n{answer}\nq\n", capture_output=True,
                              text=True, timeout=120, check=True).stdout
    cancelled = session("n")
    assert "Batch assembly preview" in cancelled and config_path.read_bytes() == original, cancelled
    result, preview = run("assembly", "patch", "--set", "image_encoder=components:good", "--set", "tabular_encoder=components:good")
    assert result.returncode == 0 and preview["data"]["validation"]["forward"] == "passed", preview
    revision = preview["data"]["base_config_revision"]
    source_revision = preview["data"]["source_evidence"]["digest"]
    input_digest = preview["data"]["input_evidence"]["digest"]
    assert len(source_revision) == 64 and len(input_digest) == 64, preview
    factory_original = factory.read_text()
    factory.write_text(factory_original + "\n# changed after preview\n")
    result, report = run("assembly", "patch", "--set", "image_encoder=components:good", "--set", "tabular_encoder=components:good", "--apply", "--if-config-revision", revision, "--if-source-revision", source_revision)
    assert result.returncode != 0 and report["diagnostics"][0]["code"] == "SOURCE_MANIFEST_CONFLICT", report
    assert config_path.read_bytes() == original
    factory.write_text(factory_original)
    (project / "input-value.txt").write_text("2")
    result, report = run("assembly", "patch", "--set", "image_encoder=components:good", "--set", "tabular_encoder=components:good", "--apply", "--if-config-revision", revision, "--if-input-sha256", input_digest)
    assert result.returncode != 0 and report["diagnostics"][0]["code"] == "INPUT_CONTENT_MISMATCH", report
    assert config_path.read_bytes() == original
    (project / "input-value.txt").write_text("1")
    with tempfile.TemporaryDirectory(prefix="maven-patch-race-") as wrapper_dir:
        wrapper = Path(wrapper_dir) / "cli.py"
        wrapper.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if "patch" in sys.argv and "--apply" in sys.argv:
    assert "--if-source-revision" in sys.argv and "--if-input-sha256" in sys.argv
    with Path("components.py").open("a") as stream: stream.write("\\n# concurrent source change\\n")
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
        wrapper.chmod(0o755)
        raced = session("y", wrapper, pin="y")
        assert "SOURCE_MANIFEST_CONFLICT" in raced and config_path.read_bytes() == original, raced
        factory.write_text(factory_original)
    changed = json.loads(original)
    changed["concurrent_marker"] = "preserved"
    config_path.write_text(json.dumps(changed))
    result, report = run("assembly", "patch", "--set", "image_encoder=components:good", "--set", "tabular_encoder=components:good", "--apply", "--if-config-revision", revision)
    assert result.returncode != 0 and report["diagnostics"][0]["code"] == "PROJECT_CONFIG_CONFLICT", report
    assert json.loads(config_path.read_text())["concurrent_marker"] == "preserved"
    committed = session("y", pin="y")
    assert "Batch assembly preview" in committed and "Assembly saved" in committed, committed
    saved = json.loads(config_path.read_text())
    assert saved["model_injections"]["image_encoder"] == "components:good" and saved["model_injections"]["tabular_encoder"] == "components:good", saved
    result, report = run("assembly", "show")
    assert result.returncode == 0 and report["data"]["validation"]["forward"] == "passed", report
    result, report = run("assembly", "patch", "--remove", "tabular_encoder")
    assert result.returncode == 0 and "tabular_encoder" not in report["data"]["model_injections"], report
    assert "tabular_encoder" in json.loads(config_path.read_text())["model_injections"]
    schema = {"type":"object", "properties":{"slope":{"type":"number", "minimum":0, "maximum":1}}, "required":["slope"], "additionalProperties":False}
    result, report = run("component", "add", "local/scaled", "--kind", "layer", "--entrypoint", "components:scaled", "--params-schema", json.dumps(schema))
    assert result.returncode == 0, report
    result, report = run("assembly", "patch", "--set", "image_encoder=component:local/scaled@0.1.0", "--param", "image_encoder.slope=0.2", "--remove", "tabular_encoder")
    assert result.returncode == 0 and report["data"]["model_injections"]["image_encoder"]["params"]["slope"] == 0.2, report
    result, report = run("assembly", "patch", "--set", "image_encoder=components:good", "--remove", "image_encoder")
    assert result.returncode != 0 and report["diagnostics"][0]["code"] == "ASSEMBLY_PATCH_INVALID", report
    result, report = run("assembly", "patch", "--set", "image_encoder=components:good", "--apply")
    assert result.returncode != 0 and report["diagnostics"][0]["code"] == "PROJECT_CONFIG_GUARD_REQUIRED", report
    current = starter.read_text()
    old = '    result["kwargs"]["image"] *= float(Path("input-value.txt").read_text())'
    new = '    path = Path("input-value.txt")\n    value = float(path.read_text())\n    path.write_text(str(value + 1))\n    result["kwargs"]["image"] *= value'
    assert old in current
    starter.write_text(current.replace(old, new))
    (project / "batch_adapter.py").write_text("import model_starter\ndef batch(): return model_starter.batch()\n")
    updated = json.loads(config_path.read_text())
    updated["batch"] = "batch_adapter:batch"
    config_path.write_text(json.dumps(updated))
    (project / "input-value.txt").write_text("1")
    snapshot = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
        input=f"\na\npatch+snapshot\n{inputs}\n\n{{}}\ny\nq\n", capture_output=True, text=True, timeout=120, check=True)
    assert "Capturing input snapshot" in snapshot.stdout and "Temporary input snapshot removed" in snapshot.stdout and "Assembly saved" in snapshot.stdout, snapshot.stdout
    match = re.search(r"Temporary file: ([^\s]+)", snapshot.stdout)
    assert match and not Path(match.group(1)).exists(), snapshot.stdout
    assert float((project / "input-value.txt").read_text()) == 2, snapshot.stdout
    print("Changing batch is captured once and replayed for atomic assembly; temporary input is removed")
    print("Two broken seams are repaired together; cancellation and config race preserve original state")
