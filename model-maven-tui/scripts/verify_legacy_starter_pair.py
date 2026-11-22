#!/usr/bin/env python3
"""Exercise actionable diagnostics with an actual previously generated starter."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
backend = root.parent / "model-maven"
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(backend / "target/debug/m-model"))
fixture = backend / "docs/acceptance/results/journey-a-20261121/model_starter.py"
with tempfile.TemporaryDirectory(prefix="maven-old-starter-") as directory:
    project = Path(directory)
    source = project / "model_starter.py"
    shutil.copyfile(fixture, source)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    def run(*args, ok=True):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                              env=env, text=True, capture_output=True, timeout=120)
        report = json.loads(proc.stdout)
        assert (proc.returncode == 0) == ok, report
        return report
    def session(answers):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input=answers, text=True, capture_output=True,
                              timeout=180, check=True).stdout
    run("init", "--model", "model_starter:model", "--batch", "model_starter:batch")
    config = project / ".model-maven/project.json"
    original, original_config = source.read_bytes(), config.read_bytes()
    for args, option in [(("assembly", "encoder-output", "image_encoder", "native", "--dry-run"), "encoder_output_modes"),
                         (("assembly", "route", "image_encoder", "tabular", "--dry-run"), "input_routes")]:
        report = run(*args, ok=False)
        diagnostic = report["diagnostics"][0]
        assert diagnostic["code"] == "MODEL_ASSEMBLY_OPTIONS_UNSUPPORTED", report
        assert option in json.dumps(diagnostic, ensure_ascii=False), report
        assert "新版骨架" in diagnostic["hint"], report
    for answers in ("\na\nencoder-output\nimage_encoder\nnative\nq\n",
                    "\na\nroute\nimage_encoder\ntabular\nq\n"):
        output = session(answers)
        assert "MODEL_ASSEMBLY_OPTIONS_UNSUPPORTED" in output and "新版骨架" in output, output
    assert source.read_bytes() == original and config.read_bytes() == original_config
    # A dynamic factory that accepts and ignores keywords cannot establish the
    # starter's construction-time route or output-mode declaration.
    text = source.read_text().replace(
        "def model(image_encoder=None, tabular_encoder=None, fusion=None, head=None):",
        "def model(image_encoder=None, tabular_encoder=None, fusion=None, head=None, **options):")
    assert text != original.decode()
    source.write_text(text)
    for args in (("assembly", "encoder-output", "image_encoder", "native", "--dry-run"),
                 ("assembly", "route", "image_encoder", "tabular", "--dry-run")):
        result = run(*args, ok=False)
        assert result["diagnostics"][0]["code"] == "MODEL_ASSEMBLY_OPTIONS_UNCONFIRMED", result
    assert config.read_bytes() == original_config
    print("Actual legacy starter: CLI/TUI explain unsupported options without writes; ignored **kwargs is rejected")
