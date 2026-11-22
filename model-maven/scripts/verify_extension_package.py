#!/usr/bin/env python3
"""Build and exercise the example extension through the public CLI workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run_process(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"命令失败（{result.returncode}）：{' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


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
        description="用独立 wheel 验收注册 Layer 的发现、参数化注入、替换、训练和便携恢复。"
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
        default=repository / "examples/extensions/silu-component",
        help="独立扩展示例目录",
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

    with tempfile.TemporaryDirectory(prefix="model-maven-extension-acceptance-") as temporary:
        root = Path(temporary)
        package_copy = root / "extension-source"
        wheels = root / "wheels"
        site_packages = root / "site-packages"
        wheels.mkdir()
        site_packages.mkdir()
        shutil.copytree(package, package_copy)

        # Prove discovery reads the sidecar without importing the factory module.
        layer_module = package_copy / "src/model_maven_silu/layer.py"
        layer_module.write_text(
            'from pathlib import Path\n'
            'Path(__file__).with_name("imported.marker").write_text("loaded")\n'
            + layer_module.read_text(encoding="utf-8"),
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
                str(package_copy),
            ],
            cwd=root,
        )
        wheel_files = list(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise RuntimeError(f"预期构建一个 wheel，实际得到 {len(wheel_files)} 个。")
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
                str(wheel_files[0]),
            ],
            cwd=root,
        )

        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(site_packages), existing_pythonpath) if value
        )
        project = root / "project"
        project.mkdir()
        (project / "model.py").write_text(
            "from collections import OrderedDict\n"
            "import torch\n"
            "from torch import nn\n\n"
            "class ExampleModel(nn.Module):\n"
            "    def __init__(self, activation):\n"
            "        super().__init__()\n"
            "        self.encoder = nn.Sequential(OrderedDict([\n"
            "            ('proj', nn.Linear(8, 8)), ('relu', activation)\n"
            "        ]))\n\n"
            "    def forward(self, x):\n"
            "        return self.encoder(x)\n\n"
            "def build_model(encoder):\n"
            "    return ExampleModel(encoder)\n\n"
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
            binary,
            args.python,
            project,
            environment,
            "component",
            "discover",
            "--path",
            str(site_packages),
        )
        if discovered["data"]["found"] != 1:
            raise RuntimeError("CLI 没有从独立 wheel 发现唯一示例组件。")
        marker = site_packages / "model_maven_silu/imported.marker"
        if marker.exists():
            raise RuntimeError("component discover 导入了扩展工厂模块；发现必须只读静态清单。")

        tested = run_cli(
            binary,
            args.python,
            project,
            environment,
            "component",
            "test",
            "example/silu@0.1.0",
            "--param",
            "inplace=false",
        )
        if tested["data"]["status"] != "passed" or not marker.is_file():
            raise RuntimeError("扩展运行时符合性检查没有导入并构造该 Layer。")

        initialized = run_cli(
            binary,
            args.python,
            project,
            environment,
            "init",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
            "--inject",
            "encoder=component:example/silu@0.1.0",
            "--inject-param",
            "encoder.inplace=false",
        )
        injection = initialized["data"]["config"]["model_injections"]["encoder"]
        if (
            injection["component_ref"]["id"] != "example/silu"
            or injection["params"].get("inplace") is not False
        ):
            raise RuntimeError("项目默认值没有保存已解析的 Layer 参数和组件版本锁。")

        injected_probe = run_cli(
            binary, args.python, project, environment, "model", "probe"
        )
        if not any(
            item.get("slot") == "encoder"
            and item.get("component_ref", {}).get("id") == "example/silu"
            and item.get("registered_paths") == ["model.encoder.relu"]
            for item in injected_probe["data"]["origin"]["injection_evidence"]
        ):
            raise RuntimeError("probe 没有确认外部组件接缝进入模型注册树。")

        edited = run_cli(
            binary,
            args.python,
            project,
            environment,
            "layer",
            "replace",
            "model.encoder.relu",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
            "--component",
            "example/silu@0.1.0",
            "--param",
            "inplace=false",
        )
        if not edited["data"]["committed"] or edited["data"]["validation"]["forward"] != "passed":
            raise RuntimeError("独立扩展 Layer 没有通过真实候选前向并持久化为变体。")

        reloaded = run_cli(
            binary,
            args.python,
            project,
            environment,
            "--variant",
            "working",
            "model",
            "probe",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
        )
        output_shape = reloaded["data"]["outputs"].get("shape")
        layer_types = {call["module_type"] for call in reloaded["data"]["calls"]}
        if output_shape != [2, 8] or "SiLU" not in layer_types:
            raise RuntimeError("新 CLI 进程未能重载扩展变体并通过代表性批次。")

        trained = run_cli(
            binary,
            args.python,
            project,
            environment,
            "--variant",
            "working",
            "train",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
            "--step",
            "model:training_step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        )
        export_directory = root / "portable-export"
        exported = run_cli(
            binary,
            args.python,
            project,
            environment,
            "export",
            str(export_directory),
        )
        export_manifest = json.loads(
            (export_directory / "manifest.json").read_text(encoding="utf-8")
        )
        if exported["data"]["extension_package_count"] != 1:
            raise RuntimeError("训练导出没有携带注册 Layer 的 extension distribution。")
        if len(export_manifest["extension_packages"]) != 1:
            raise RuntimeError("导出清单中的扩展包记录数量不正确。")

        # A changed original installation must not shadow the hash-verified bundle.
        original_layer = site_packages / "model_maven_silu/layer.py"
        original_layer.write_text(
            original_layer.read_text(encoding="utf-8") + "\n# changed after export\n",
            encoding="utf-8",
        )
        restore_with_changed_install = subprocess.run(
            [args.python, str(export_directory / "restore_model.py")],
            cwd=export_directory,
            env=environment,
            text=True,
            capture_output=True,
        )
        if restore_with_changed_install.returncode != 0:
            raise RuntimeError(
                "恢复流程使用了被修改的原始扩展，而不是导出的副本。\n"
                f"stdout:\n{restore_with_changed_install.stdout}\n"
                f"stderr:\n{restore_with_changed_install.stderr}"
            )

        # The original wheel installation must also be unnecessary after removal.
        shutil.rmtree(site_packages)
        restore_environment = os.environ.copy()
        remaining_pythonpath = [
            value
            for value in restore_environment.get("PYTHONPATH", "").split(os.pathsep)
            if value and Path(value).resolve() != site_packages.resolve()
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
                "删除原始 extension 安装后，导出工程无法恢复模型权重。\n"
                f"stdout:\n{restore.stdout}\nstderr:\n{restore.stderr}"
            )

        package_record = export_manifest["extension_packages"][0]
        relative_file, expected_file_hash = next(iter(package_record["files"].items()))
        bundled_file = export_directory / package_record["bundle_path"] / relative_file
        bundled_file.write_bytes(bundled_file.read_bytes() + b"tamper")
        tampered_restore = subprocess.run(
            [args.python, str(export_directory / "restore_model.py")],
            cwd=export_directory,
            env=restore_environment,
            text=True,
            capture_output=True,
        )
        if tampered_restore.returncode == 0 or "SHA-256" not in tampered_restore.stderr:
            raise RuntimeError("恢复流程没有拒绝被篡改的扩展包文件。")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "wheel": wheel_files[0].name,
                    "discovered_components": discovered["data"]["found"],
                    "factory_check": tested["data"]["status"],
                    "registered_injection": "passed",
                    "layer_replace_forward": edited["data"]["validation"]["forward"],
                    "reloaded_output_shape": output_shape,
                    "reloaded_layer": "SiLU",
                    "formal_training_steps": trained["data"]["completed_steps"],
                    "exported_extension_files": exported["data"]["extension_file_count"],
                    "offline_export_restore": "passed",
                    "changed_original_did_not_shadow_bundle": "passed",
                    "tampered_bundle_rejected": "passed",
                    "limits": [
                        "验证本机 wheel 的发现、Layer 替换、正式训练、包文件归档与删除原安装后的 checkpoint 恢复。",
                        "不验证任意扩展包的安全性、传递依赖、Provider 状态符合性或多平台 ABI 兼容。",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"extension acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
