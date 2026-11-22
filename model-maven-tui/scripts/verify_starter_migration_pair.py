#!/usr/bin/env python3
"""Migrate an actual older starter through the real CLI and TUI."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
backend = root.parent / "model-maven"
cli = os.environ.get("MODEL_MAVEN_CLI", str(backend / "target/debug/m-model"))
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
fixture = backend / "docs/acceptance/results/journey-a-20261121/model_starter.py"
with tempfile.TemporaryDirectory(prefix="maven-starter-migration-") as directory:
    project = Path(directory)
    source = project / "model_starter.py"
    shutil.copyfile(fixture, source)
    starter = source.read_text().replace("class Model(nn.Module):", "_user_global = torch.rand(5)\n\nclass Model(nn.Module):")
    old_head = "        self.head = head if head is not None else nn.Linear(53, 3)"
    assert old_head in starter
    starter = starter.replace(old_head, old_head + "\n        with open('construction.log', 'a') as log:\n            log.write(float(self.head.weight.flatten()[0]).hex() + '\\n')")
    source.write_text(starter + "\n# user-owned code\ndef user_helper(): return 17\n")
    (project / "fixture_component.py").write_text("import torch\nfrom torch import nn\ndef encoder():\n    torch.rand(7)\n    return nn.Identity()\n")
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    def run(*args, ok=True):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, text=True, capture_output=True, timeout=120)
        report = json.loads(result.stdout)
        assert (result.returncode == 0) == ok, report
        return report
    def session(answers):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input=answers, text=True, capture_output=True,
                              timeout=180, check=True).stdout
    run("init", "--model", "model_starter:model", "--batch", "model_starter:batch",
        "--inject", "image_encoder=fixture_component:encoder")
    original = source.read_bytes()
    config = (project / ".model-maven/project.json").read_bytes()
    missing = run("layer", "migrate-starter", "--target", "source", ok=False)
    assert missing["diagnostics"][0]["code"] == "SOURCE_PLAN_REQUIRED", missing
    preview = run("layer", "migrate-starter", "--target", "source", "--dry-run", "--require-forward")
    assert preview["data"]["validation"]["forward"] == "passed" and source.read_bytes() == original
    constructions = (project / "construction.log").read_text().splitlines()
    assert len(constructions) == 1, constructions
    digest = preview["data"]["source_plan"]["digest"]
    source.write_bytes(original + b"# concurrent edit\n")
    conflict = run("--if-source-plan", digest, "layer", "migrate-starter", "--target", "source", "--require-forward", ok=False)
    assert conflict["diagnostics"][0]["code"] == "SOURCE_PLAN_CONFLICT", conflict
    assert source.read_bytes().endswith(b"# concurrent edit\n")
    source.write_bytes(original)
    cancelled = session("\nh\nmigrate-starter\nn\nq\n")
    assert "Source preview" in cancelled and source.read_bytes() == original, cancelled
    committed = session("\nh\nmigrate-starter\napply\nq\n")
    assert "Post-save verification" in committed and '"status":"passed"' in committed, committed
    migrated = source.read_bytes()
    assert b"def user_helper(): return 17" in migrated and b"def _m_model_encode" in migrated
    assert (project / ".model-maven/project.json").read_bytes() == config
    undo = session("\nh\nundo\napply\nq\n")
    assert "separate forward recheck" in undo and source.read_bytes() == original, undo
    redo = session("\nh\nredo\napply\nq\n")
    assert "separate forward recheck" in redo and source.read_bytes() == migrated, redo
    native = run("assembly", "encoder-output", "tabular_encoder", "native", "--dry-run")
    assert native["data"]["validation"]["forward"] == "passed", native
    native_saved = session("\na\nencoder-output\ntabular_encoder\nnative\ny\ny\nq\n")
    assert json.loads((project / ".model-maven/project.json").read_text())["model_params"]["encoder_output_modes"]["tabular_encoder"] == "native", native_saved
    assert run("model", "probe")["status"] == "success"
    constructions = (project / "construction.log").read_text().splitlines()
    assert len(set(constructions)) == 1, constructions
    print("Actual old starter: preview, cancel, source conflict, migrate, recheck, undo/redo, and native output save work")

with tempfile.TemporaryDirectory(prefix="maven-nested-starter-") as directory:
    project = Path(directory)
    package = project / "custom_model"
    package.mkdir()
    (package / "__init__.py").write_text("# package\n")
    nested_source = package / "starter.py"
    shutil.copyfile(fixture, nested_source)
    (project / "data.py").write_text("from custom_model.starter import batch as old_batch\ndef batch(): return old_batch()\n")
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    def command(*args, ok=True):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, text=True, capture_output=True, timeout=120)
        report = json.loads(result.stdout)
        assert (result.returncode == 0) == ok, report
        return report
    command("init", "--model", "custom_model.starter:model", "--batch", "data:batch")
    output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                            env=env, input="\nh\nmigrate-starter+snapshot\napply\nq\n",
                            text=True, capture_output=True, timeout=180, check=True).stdout
    assert "Post-save verification" in output and '"status":"passed"' in output, output
    assert b"def _m_model_encode" in nested_source.read_bytes()
    assert command("model", "probe")["status"] == "success"
    command("layer", "undo", "--target", "source")
    original = nested_source.read_bytes()
    (package / "__init__.py").write_text("from . import starter\n")
    rejected = command("layer", "migrate-starter", "--target", "source", "--dry-run", ok=False)
    assert rejected["diagnostics"][0]["code"] == "STARTER_MIGRATION_UNSUPPORTED", rejected
    assert nested_source.read_bytes() == original
    shutil.copyfile(fixture, project / "alternate.py")
    (package / "__init__.py").write_text('''import importlib.util, sys
from pathlib import Path
class Redirect:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "custom_model.starter":
            return importlib.util.spec_from_file_location(fullname, Path(__file__).parent.parent / "alternate.py")
sys.meta_path.insert(0, Redirect())
''')
    redirected = command("layer", "migrate-starter", "--target", "source", "--dry-run", ok=False)
    assert redirected["diagnostics"][0]["code"] == "STARTER_MIGRATION_UNSUPPORTED", redirected
    assert nested_source.read_bytes() == original
    print("Nested starter and external batch migrate with input snapshot; eager import and redirected loader are rejected without writes")
