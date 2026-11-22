#!/usr/bin/env python3
"""Build and exercise an independent Layer Edit Provider wheel end to end."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run_process(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"命令失败（{result.returncode}）：{' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_cli(binary: Path, python: str, cwd: Path, env: dict[str, str], *args: str) -> dict:
    command = [str(binary), "--python", python, "--json", *args]
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"CLI 未返回单个 JSON 报告：{' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from error
    if result.returncode != 0 or report.get("status") != "success":
        raise RuntimeError(
            f"CLI 操作失败：{' '.join(command)}\n"
            f"{json.dumps(report, ensure_ascii=False, indent=2)}\nstderr:\n{result.stderr}"
        )
    return report


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="用隔离 wheel 验收独立 Layer Edit Provider 的发现、验证、编辑与导出。"
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=repository / "target/debug/m-model",
        help="已构建的 m-model 二进制（默认 target/debug/m-model）",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="包含 PyTorch、pip、setuptools 和 wheel 的 Python 解释器",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=repository / "examples/extensions/silu-provider",
        help="独立 Provider 扩展示例目录",
    )
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve()
    package = args.package.expanduser().resolve()
    if not binary.is_file():
        parser.error(f"找不到 CLI 二进制：{binary}；先运行 cargo build。")
    if not package.is_dir():
        parser.error(f"找不到扩展包目录：{package}")
    build_tools = subprocess.run(
        [
            args.python,
            "-c",
            "import importlib.metadata as m; print(m.version('setuptools')); print(m.version('wheel'))",
        ],
        text=True,
        capture_output=True,
    )
    if build_tools.returncode != 0:
        parser.error(
            f"{args.python} 环境需要 pip、setuptools 和 wheel；请先在该环境中安装构建工具。\n"
            f"{build_tools.stderr.strip()}"
        )

    with tempfile.TemporaryDirectory(prefix="model-maven-provider-acceptance-") as temporary:
        root = Path(temporary)
        package_copy = root / "extension-source"
        helper_copy = root / "helper-source"
        wheels = root / "wheels"
        site_packages = root / "site-packages"
        wheels.mkdir()
        site_packages.mkdir()
        shutil.copytree(package, package_copy)
        shutil.copytree(repository / "examples/extensions/silu-helper", helper_copy)

        provider_module = package_copy / "src/model_maven_silu_provider/provider.py"
        provider_module.write_text(
            'from pathlib import Path\n'
            'Path(__file__).with_name("imported.marker").write_text("loaded")\n'
            + provider_module.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        run_process(
            [
                args.python,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(wheels),
                str(helper_copy),
            ],
            cwd=root,
        )
        run_process(
            [
                args.python,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(wheels),
                str(package_copy),
            ],
            cwd=root,
        )
        wheel_files = list(wheels.glob("*.whl"))
        if len(wheel_files) != 2:
            raise RuntimeError(f"预期构建 Provider 与 helper 两个 wheel，实际得到 {len(wheel_files)} 个。")
        run_process(
            [
                args.python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(site_packages),
                *(str(wheel) for wheel in wheel_files),
            ],
            cwd=root,
        )

        environment = os.environ.copy()
        old_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(site_packages), old_pythonpath) if value
        )
        project = root / "project"
        project.mkdir()
        (project / "model.py").write_text(
            "from collections import OrderedDict\n"
            "import torch\n"
            "from torch import nn\n\n"
            "class ExampleModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.encoder = nn.Sequential(OrderedDict([\n"
            "            ('projection', nn.Linear(8, 8)), ('relu', nn.ReLU())\n"
            "        ]))\n\n"
            "    def forward(self, values):\n"
            "        return self.encoder(values)\n\n"
            "def build_model():\n"
            "    return ExampleModel()\n\n"
            "def example_batch():\n"
            "    return {'args': [torch.randn(2, 8)], 'kwargs': {}, "
            "'loss_inputs': {'target': torch.randn(2, 8)}}\n\n"
            "def training_step(model, batch, context):\n"
            "    output = model(*batch['args'], **batch['kwargs'])\n"
            "    loss = (output - batch['loss_inputs']['target']).square().mean()\n"
            "    return {'loss': loss}\n",
            encoding="utf-8",
        )

        discovered = run_cli(
            binary, args.python, project, environment,
            "component", "discover", "--path", str(site_packages),
        )
        if (
            discovered["data"]["found"] != 1
            or len(discovered["data"].get("registered", [])) != 1
        ):
            raise RuntimeError(
                "CLI 没有从独立 Provider wheel 发现并注册唯一组件：\n"
                f"{json.dumps(discovered['data'], ensure_ascii=False, indent=2)}"
            )
        marker = site_packages / "model_maven_silu_provider/imported.marker"
        if marker.exists():
            raise RuntimeError("component discover 导入了 Provider；发现必须只读静态清单。")

        tested = run_cli(
            binary, args.python, project, environment,
            "component", "test", "example/append-silu@0.1.0", "--param", 'name="silu"',
        )
        if (
            tested["data"]["coverage"] != "complete_for_fixture_expectations"
            or not tested["data"]["activation_eligible"]
            or not marker.is_file()
        ):
            raise RuntimeError("Provider fixture 没有完成显式断言或 Provider 未被加载。")

        edited = run_cli(
            binary, args.python, project, environment,
            "layer", "custom", "example/append-silu@0.1.0",
            "--payload", '{"name":"silu"}',
            "--model", "model:build_model", "--batch", "model:example_batch",
        )
        if (
            not edited["data"]["committed"]
            or edited["data"]["validation"]["forward"] != "passed"
        ):
            raise RuntimeError("已通过 fixture 检查的 Provider 没有完成真实模型编辑。")

        undone = run_cli(binary, args.python, project, environment, "layer", "undo")
        if not undone["data"]["committed"]:
            raise RuntimeError("Provider 变体编辑未能撤销。")
        run_cli(binary, args.python, project, environment, "layer", "redo")

        reloaded = run_cli(
            binary, args.python, project, environment,
            "--variant", "working", "model", "probe",
            "--model", "model:build_model", "--batch", "model:example_batch",
        )
        output_shape = reloaded["data"]["outputs"].get("shape")
        layer_types = {call["module_type"] for call in reloaded["data"]["calls"]}
        if output_shape != [2, 8] or "SiLU" not in layer_types:
            raise RuntimeError("新 CLI 进程未能重放 Provider 变体并通过真实批次。")

        trained = run_cli(
            binary, args.python, project, environment,
            "--variant", "working", "train",
            "--model", "model:build_model", "--batch", "model:example_batch",
            "--step", "model:training_step", "--steps", "1", "--checkpoint-every", "1",
        )
        export_directory = root / "portable-export"
        exported = run_cli(binary, args.python, project, environment, "export", str(export_directory))
        manifest_path = export_directory / "manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        packages = {record["distribution"]: record for record in manifest["extension_packages"]}
        helper_record = packages.get("model-maven-silu-helper")
        if (
            exported["data"]["extension_package_count"] != 1
            or exported["data"]["dependency_package_count"] != 1
            or len(manifest["extension_packages"]) != 2
            or helper_record is None
            or helper_record["role"] != "runtime_dependency"
            or not helper_record["required_by"]
            or manifest["runtime_requirements"][0]["distribution"].lower() != "torch"
        ):
            raise RuntimeError(
                "训练导出没有携带 Provider 与 helper 传递依赖、依赖来源和 Torch 外部版本约束：\n"
                f"{json.dumps(manifest.get('extension_packages'), ensure_ascii=False, indent=2)}"
            )

        mismatched_manifest = json.loads(manifest_text)
        mismatched_manifest["runtime_requirements"][0]["version"] = "0.0.0-model-maven-test-mismatch"
        manifest_path.write_text(
            json.dumps(mismatched_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        runtime_mismatch = subprocess.run(
            [args.python, str(export_directory / "restore_model.py")],
            cwd=export_directory,
            env=environment,
            text=True,
            capture_output=True,
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        if runtime_mismatch.returncode == 0 or "运行时依赖版本不匹配" not in runtime_mismatch.stderr:
            raise RuntimeError(
                "恢复流程没有拒绝与导出记录不匹配的 PyTorch 版本。\n"
                f"stdout:\n{runtime_mismatch.stdout}\nstderr:\n{runtime_mismatch.stderr}"
            )

        installed_provider = site_packages / "model_maven_silu_provider/provider.py"
        installed_provider.write_text(
            installed_provider.read_text(encoding="utf-8") + "\n# changed after export\n",
            encoding="utf-8",
        )
        installed_helper = site_packages / "model_maven_silu_helper/__init__.py"
        installed_helper.write_text(
            installed_helper.read_text(encoding="utf-8") + "\n# changed after export\n",
            encoding="utf-8",
        )
        changed_restore = subprocess.run(
            [
                args.python,
                "-c",
                "import runpy, sys; runpy.run_path(sys.argv[1]); "
                "print(sys.modules['model_maven_silu_helper'].__file__)",
                str(export_directory / "restore_model.py"),
            ],
            cwd=export_directory,
            env=environment,
            text=True,
            capture_output=True,
        )
        if changed_restore.returncode != 0:
            raise RuntimeError(
                "恢复流程没有优先使用哈希校验的 Provider bundle 副本。\n"
                f"stdout:\n{changed_restore.stdout}\nstderr:\n{changed_restore.stderr}"
            )
        if str(export_directory / "vendor/packages") not in changed_restore.stdout:
            raise RuntimeError(
                "恢复流程没有优先使用哈希校验的 helper bundle 副本。\n"
                f"stdout:\n{changed_restore.stdout}"
            )

        shutil.rmtree(site_packages)
        restore_environment = os.environ.copy()
        remaining_pythonpath = [
            item for item in restore_environment.get("PYTHONPATH", "").split(os.pathsep)
            if item and Path(item).resolve() != site_packages.resolve()
        ]
        if remaining_pythonpath:
            restore_environment["PYTHONPATH"] = os.pathsep.join(remaining_pythonpath)
        else:
            restore_environment.pop("PYTHONPATH", None)
        restore = subprocess.run(
            [args.python, str(export_directory / "restore_model.py")],
            cwd=export_directory,
            env=restore_environment,
            text=True,
            capture_output=True,
        )
        if restore.returncode != 0 or str(trained["data"]["run_id"]) not in restore.stdout:
            raise RuntimeError(
                "删除原始 Provider 安装后，导出工程无法恢复模型。\n"
                f"stdout:\n{restore.stdout}\nstderr:\n{restore.stderr}"
            )

        package_record = manifest["extension_packages"][0]
        relative_file = next(iter(package_record["files"]))
        bundle_file = export_directory / package_record["bundle_path"] / relative_file
        bundle_file.write_bytes(bundle_file.read_bytes() + b"tamper")
        tampered_restore = subprocess.run(
            [args.python, str(export_directory / "restore_model.py")],
            cwd=export_directory,
            env=restore_environment,
            text=True,
            capture_output=True,
        )
        if tampered_restore.returncode == 0 or "SHA-256" not in tampered_restore.stderr:
            raise RuntimeError("恢复流程没有拒绝被篡改的 Provider bundle 文件。")

        print(json.dumps({
            "status": "passed",
            "wheel": wheel_files[0].name,
            "discovered_providers": discovered["data"]["found"],
            "fixture_coverage": tested["data"]["coverage"],
            "real_model_edit": edited["data"]["validation"]["forward"],
            "variant_undo_redo": "passed",
            "reloaded_output_shape": output_shape,
            "reloaded_layer": "SiLU",
            "formal_training_steps": trained["data"]["completed_steps"],
            "exported_provider_files": exported["data"]["extension_file_count"],
            "offline_export_restore": "passed",
            "external_runtime_version_mismatch_rejected": "passed",
            "dependency_closure": [
                record["distribution"] for record in manifest["extension_packages"]
            ],
            "changed_original_provider_and_dependency_did_not_shadow_bundle": "passed",
            "tampered_bundle_rejected": "passed",
            "limits": [
                "覆盖一个受支持 Python/PyTorch 环境中的单一 Provider wheel、一个纯 Python PEP 508 helper 依赖和显式 fixture payload。",
                "不证明任意依赖图、模型/输入正确性、恶意扩展安全性、系统库闭包或跨平台 ABI 兼容。",
            ],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Provider extension acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
