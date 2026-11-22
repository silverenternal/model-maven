#!/usr/bin/env python3
"""Exercise installed Linux release binaries outside both source trees.

PyTorch is inherited from system site-packages; this does not verify dependency
provisioning, distribution packages, or an upgrade from an older release.
"""
import os
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from verify_integration import identity, repository

tui_root = Path(__file__).resolve().parent.parent
cli_root = tui_root.parent / "model-maven"
evidence = Path(os.environ["MODEL_MAVEN_INSTALL_EVIDENCE"]).resolve() if os.environ.get("MODEL_MAVEN_INSTALL_EVIDENCE") else None
if evidence:
    evidence.mkdir(parents=True, exist_ok=False)
    source_before = {"backend": repository(cli_root, evidence), "tui": repository(tui_root, evidence)}


def run(args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    if evidence:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)
        with (evidence / "commands.log").open("a") as stream:
            stream.write("+ " + " ".join(map(str, args)) + "\n" + result.stdout + "\n")
        if result.returncode:
            print(result.stdout, flush=True)
            raise subprocess.CalledProcessError(result.returncode, args)
    else:
        subprocess.run(args, check=True, **kwargs)


with tempfile.TemporaryDirectory(prefix="maven-install-pair-") as temp:
    temp = Path(temp)
    prefix = temp / "install"
    project = temp / "project"
    project.mkdir()
    venv = temp / "python"
    run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
    python = venv / "bin/python3"
    run([str(python), "-c", "import sys, torch; print(sys.version.split()[0], torch.__version__)"])
    for root in (cli_root, tui_root):
        env = dict(os.environ, CARGO_TARGET_DIR=str(root / "target"))
        run(["cargo", "install", "--path", str(root), "--root", str(prefix),
             "--locked", "--offline", "--no-track"], env=env)

    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "MODEL_MAVEN_CLI"):
        env.pop(key, None)
    env["PATH"] = os.pathsep.join((str(prefix / "bin"), str(venv / "bin"), env["PATH"]))
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(temp / "preferences.json")
    cli = prefix / "bin/m-model"
    tui = prefix / "bin/m-model-tui"
    binary_identity = {"cli": identity(cli), "tui": identity(tui)} if evidence else None
    run([str(cli), "--version"], cwd=project, env=env)
    run([str(tui), "--version"], cwd=project, env=env)
    # With no --cli or MODEL_MAVEN_CLI, the TUI must discover the installed CLI on PATH.
    (project / "fixture.py").write_text(
        "import torch\nfrom torch import nn\n"
        "def model(): return nn.Linear(4, 2)\n"
        "def batch(): return torch.ones(2, 4)\n"
    )
    result = subprocess.run([str(tui), "--line-mode", "--lang", "en"], cwd=project,
                            env=env, input="\nn\nfixture:model\nfixture:batch\ny\nq\n", text=True, capture_output=True,
                            timeout=60, check=True)
    assert "Project setup preview" in result.stdout, (result.stdout, result.stderr)
    if evidence:
        (evidence / "path-discovery.txt").write_text(result.stdout)
    print("TUI PATH discovery session completed", flush=True)

    # Run the same end-to-end acceptance against installed release binaries.
    env["MODEL_MAVEN_CLI"] = str(cli)
    env["MODEL_MAVEN_TUI_BIN"] = str(tui)
    if evidence:
        env["MODEL_MAVEN_JOURNEY_EVIDENCE"] = str(evidence / "journey_a")
        env["MODEL_MAVEN_JOURNEY_B_EVIDENCE"] = str(evidence / "journey_b")
    for journey in ("verify_journey_a.py", "verify_journey_b.py"):
        run([str(python), str(tui_root / "scripts" / journey)], cwd=project, env=env)

    # Exercise switching and reverting a versioned installation. The two copies
    # are identical; this is not evidence of compatibility with an older release.
    versions = temp / "versions"
    for name in ("previous", "candidate"):
        shutil.copytree(prefix / "bin", versions / name / "bin")
    current = temp / "current"
    current.symlink_to(versions / "previous", target_is_directory=True)
    for name in ("candidate", "previous"):
        replacement = temp / "current.next"
        replacement.symlink_to(versions / name, target_is_directory=True)
        os.replace(replacement, current)
        run([str(current / "bin/m-model"), "--version"], cwd=project, env=env)
        run([str(current / "bin/m-model-tui"), "--version"], cwd=project, env=env)
    if evidence:
        source_after = {"backend": repository(cli_root, evidence), "tui": repository(tui_root, evidence)}
        unchanged = all(source_before[key]["content_identity"]["status"] == "available"
                        and source_before[key]["content_identity"]["sha256"] == source_after[key]["content_identity"]["sha256"]
                        for key in source_before)
        manifest = {"status": "passed" if unchanged else "failed_source_changed", "platform": platform.platform(),
                    "python_and_torch": subprocess.check_output([str(python), "-c", "import sys, torch; print(sys.version.split()[0], torch.__version__)"], text=True).strip(),
                    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
                    "installed_binaries": binary_identity, "source_before": source_before, "source_after": source_after,
                    "limits": ["system-site PyTorch; no dependency provisioning", "local Linux only", "previous and candidate binaries are identical", "legacy behavior simulated by wrappers"]}
        (evidence / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        assert unchanged, "source observation changed during installed release validation"
    print("PASS: installed release journeys and local switch/rollback", flush=True)
