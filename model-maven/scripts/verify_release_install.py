#!/usr/bin/env python3
"""Verify the packaged CLI and optional SDK work outside a source checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command, cwd=cwd, env=env, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}: {' '.join(command)}")
    return completed.stdout


def verify_sdk(temp: Path) -> None:
    wheelhouse = temp / "sdk-wheels"
    wheelhouse.mkdir()
    sdk_source = temp / "sdk-source"
    shutil.copytree(ROOT / "python/model_maven_sdk", sdk_source)
    run([
        sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
        "--wheel-dir", str(wheelhouse), str(sdk_source),
    ], cwd=temp)
    wheels = list(wheelhouse.glob("model_maven_sdk-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one SDK wheel, found {len(wheels)}")
    environment = temp / "sdk-venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run([str(python), "-m", "pip", "install", "--no-index", str(wheels[0])], cwd=temp)
    run([
        str(python), "-c",
        "from model_maven_sdk import ComponentManifest, Evaluator, EvaluatorContext; "
        "print(ComponentManifest.__name__, Evaluator.__name__, EvaluatorContext.__name__)",
    ], cwd=temp)


def verify_evaluation_preview(binary: Path, python: str, temp: Path, env: dict[str, str]) -> None:
    """Exercise evaluator-owned preview through the extracted, installed CLI."""
    project = temp / "evaluation-preview-project"
    project.mkdir()
    (project / "release_evaluation_fixture.py").write_text(
        """import torch
from torch import nn

def build_model():
    return nn.Linear(2, 2)

def batch():
    return torch.ones(2, 2), torch.tensor([0, 1])

def step(model, batch, context):
    values, target = batch
    return {"loss": nn.functional.cross_entropy(model(values), target)}

def build_data():
    return {"values": torch.ones(2, 2), "targets": [0, 1]}

def evaluate(model, data, context):
    logits = model(data["values"])
    context.report_samples(len(data["targets"]))
    context.report_prediction("sample-0", logits[0].tolist(), data["targets"][0])
    return {"metrics": {"fixture/accuracy": 0.5}}
""",
        encoding="utf-8",
    )

    def json_command(*arguments: str) -> dict:
        output = run(
            [str(binary), "--python", python, "--json", *arguments],
            cwd=project,
            env=env,
        )
        try:
            report = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("installed CLI did not return a JSON report") from exc
        if report.get("status") != "success":
            raise RuntimeError(f"installed CLI operation failed: {report}")
        return report

    trained = json_command(
        "train",
        "--model", "release_evaluation_fixture:build_model",
        "--batch", "release_evaluation_fixture:batch",
        "--step", "release_evaluation_fixture:step",
        "--steps", "1",
        "--checkpoint-every", "1",
    )
    source_run = trained["data"]["run_id"]
    if not isinstance(source_run, str) or not source_run.startswith("run-"):
        raise RuntimeError("installed CLI did not create a formal source training run")

    evaluated = json_command(
        "evaluate",
        "--run", source_run,
        "--evaluator", "release_evaluation_fixture:evaluate",
        "--data", "release_evaluation_fixture:build_data",
        "--split", "validation",
        "--preview-samples", "1",
    )
    evaluation = evaluated["data"]["evaluation"]
    previews = evaluation["sample_previews"]
    if (
        previews.get("state") != "reported"
        or previews.get("limit") != 1
        or previews.get("reported") != 1
        or previews.get("samples", [{}])[0].get("sample_id") != "sample-0"
        or previews["samples"][0].get("target") != 0
        or evaluated["data"]["coverage"].get("classification_assumption") is not False
    ):
        raise RuntimeError(f"installed CLI returned an invalid typed evaluation preview: {previews}")

    evaluation_run = evaluated["data"]["run_id"]
    shown = json_command("runs", "show", evaluation_run)
    stored = shown["data"]["report"]["data"]["evaluation"]["sample_previews"]
    if stored != previews:
        raise RuntimeError("installed CLI did not persist the evaluator preview in run history")
    print("INSTALLED EVALUATION PREVIEW AND RUN HISTORY PASSED", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sdk", action="store_true", help="skip optional SDK wheel smoke")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="用于完整安装包模型训练/导出/恢复验收的 Python（需要 PyTorch）",
    )
    args = parser.parse_args()
    cargo = shutil.which("cargo")
    if not cargo:
        parser.error("cargo is required")

    with tempfile.TemporaryDirectory(prefix="model-maven-release-") as directory:
        temp = Path(directory)
        package_dir = temp / "package"
        package_dir.mkdir()
        run([
            cargo, "package", "--locked", "--allow-dirty", "--no-verify",
            "--package", "model-maven", "--target-dir", str(temp / "target"),
        ], cwd=ROOT)
        archives = list((temp / "target/package").glob("model-maven-*.crate"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one Cargo package archive, found {len(archives)}")
        with tarfile.open(archives[0], "r:gz") as archive:
            archive.extractall(package_dir, filter="data")
        crate_roots = [path for path in package_dir.iterdir() if path.is_dir()]
        if len(crate_roots) != 1:
            raise RuntimeError("Cargo archive did not contain exactly one crate root")

        install_root = temp / "install"
        run([
            cargo, "install", "--locked", "--path", str(crate_roots[0]),
            "--root", str(install_root), "--force",
        ], cwd=temp)
        binary = install_root / ("bin/m-model.exe" if os.name == "nt" else "bin/m-model")
        if not binary.is_file():
            raise RuntimeError(f"installed CLI was not found at {binary}")

        isolated = temp / "empty-project"
        isolated.mkdir()
        home = temp / "home"
        home.mkdir()
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)
        version = run([str(binary), "--version"], cwd=isolated, env=env).strip()
        manifest = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
        expected = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.MULTILINE)
        if not expected or version != f"m-model {expected.group(1)}":
            raise RuntimeError(f"installed version {version!r} does not match Cargo.toml")
        help_text = run([str(binary), "--help"], cwd=isolated, env=env)
        doctor_output = run([str(binary), "--json", "doctor"], cwd=isolated, env=env)
        try:
            doctor = json.loads(doctor_output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("installed CLI doctor did not return JSON") from exc
        if doctor.get("operation") != "doctor" or doctor.get("status") != "success":
            raise RuntimeError("installed CLI worker did not complete doctor capability discovery")
        preset_env = temp / "preset-catalog-python"
        run([args.python, "-m", "venv", "--without-pip", str(preset_env)], cwd=temp, env=env)
        preset_python = preset_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run(
            [str(preset_python), "-c", "import importlib.util; assert importlib.util.find_spec('torch') is None"],
            cwd=temp,
            env=env,
        )
        preset_output = run(
            [str(binary), "--python", str(preset_python), "--json", "data", "presets", "resnet"],
            cwd=isolated,
            env=env,
        )
        try:
            preset_report = json.loads(preset_output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("installed CLI preset catalog did not return JSON") from exc
        if (
            preset_report.get("operation") != "preset_catalog"
            or preset_report.get("status") != "success"
            or preset_report.get("data", {}).get("query") != "resnet"
            or set(preset_report.get("data", {}).get("presets", {})) != {"resnet18"}
        ):
            raise RuntimeError(f"installed CLI returned an invalid preset catalog: {preset_report}")
        print("INSTALLED PRESET CATALOG SEARCH PASSED WITHOUT PYTORCH", flush=True)
        doctor_data = doctor.get("data", {})
        implemented = doctor_data.get("implemented", [])
        not_implemented = doctor_data.get("not_implemented", [])
        if not any("cross-batch sample and violation ID deduplication" in item for item in implemented):
            raise RuntimeError("installed doctor omitted implemented checkpointed sample-ID coverage")
        if any("cross-batch unique-sample coverage" in item for item in not_implemented):
            raise RuntimeError("installed doctor still reports cross-batch sample-ID coverage as unimplemented")
        if not any("opt-in external-loop supervision" in item for item in implemented):
            raise RuntimeError("installed doctor omitted the explicit external-loop supervision protocol")
        if any("external-loop supervision protocol" in item for item in not_implemented):
            raise RuntimeError("installed doctor still reports the opt-in external-loop supervision protocol as unimplemented")
        unavailable_python = temp / "missing-python"
        unavailable = subprocess.run(
            [str(binary), "--python", str(unavailable_python), "--json", "doctor"],
            cwd=isolated,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(unavailable.stdout, end="" if unavailable.stdout.endswith("\n") else "\n")
        try:
            unavailable_report = json.loads(unavailable.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("installed CLI missing-Python path did not return JSON") from exc
        diagnostics = unavailable_report.get("diagnostics", [])
        if (
            unavailable.returncode == 0
            or unavailable_report.get("operation") != "doctor"
            or unavailable_report.get("status") != "error"
            or not diagnostics
            or diagnostics[0].get("code") != "PYTHON_UNAVAILABLE"
        ):
            raise RuntimeError("installed CLI did not fail clearly for an unavailable explicitly selected Python")
        no_python_path = temp / "no-python-path"
        no_python_path.mkdir()
        no_python_env = env.copy()
        no_python_env["PATH"] = str(no_python_path)
        for variable in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYENV_VERSION"):
            no_python_env.pop(variable, None)
        undiscoverable = subprocess.run(
            [str(binary), "--json", "doctor"],
            cwd=isolated,
            env=no_python_env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(undiscoverable.stdout, end="" if undiscoverable.stdout.endswith("\n") else "\n")
        try:
            undiscoverable_report = json.loads(undiscoverable.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("installed CLI no-interpreter PATH case did not return JSON") from exc
        diagnostics = undiscoverable_report.get("diagnostics", [])
        if (
            undiscoverable.returncode == 0
            or undiscoverable_report.get("operation") != "doctor"
            or undiscoverable_report.get("status") != "error"
            or not diagnostics
            or diagnostics[0].get("code") != "PYTHON_UNAVAILABLE"
        ):
            raise RuntimeError("installed CLI did not return PYTHON_UNAVAILABLE when PATH contains no interpreter")
        for capability in ("doctor", "export", "env", "component"):
            if capability not in help_text:
                raise RuntimeError(f"installed CLI help omits expected command {capability!r}")
        if "10 秒步骤边界 checkpoint" not in help_text:
            raise RuntimeError("installed CLI help omits the formal-training timeout checkpoint grace")
        run(
            [
                args.python,
                str(ROOT / "scripts/verify_complex_acceptance.py"),
                "--binary",
                str(binary),
                "--python",
                args.python,
            ],
            cwd=temp,
            env=env,
        )
        print("INSTALLED MODEL TRAIN/EXPORT/RESTORE WORKFLOW PASSED", flush=True)
        verify_evaluation_preview(binary, args.python, temp, env)
        if not args.skip_sdk:
            verify_sdk(temp)

    print("RELEASE INSTALL SMOKE PASSED (cargo package/install, isolated CLI, model workflow, optional SDK wheel)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        print(f"release install verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
