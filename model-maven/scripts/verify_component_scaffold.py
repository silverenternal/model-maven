#!/usr/bin/env python3
"""Exercise generated component packages through build, install, discovery, uninstall, and runtime checks."""

from __future__ import annotations

import argparse
import json
import errno
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import tempfile
import time


def run(binary: Path, python: str, root: Path, *args: str, expect_success: bool = True) -> dict:
    command = [str(binary), "--python", python, "--timeout", "600", "--json", *args]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=660)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"CLI 未返回 JSON：{' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from error
    succeeded = result.returncode == 0 and report.get("status") == "success"
    if succeeded != expect_success:
        raise RuntimeError(
            f"CLI 结果不符合预期：{' '.join(command)}\n"
            f"{json.dumps(report, ensure_ascii=False, indent=2)}\nstderr:\n{result.stderr}"
        )
    return report


def inventory(python: str) -> list[tuple[str, str]]:
    command = [
        python,
        "-c",
        "import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata.get('Name',''),d.version) for d in m.distributions() if d.metadata.get('Name') and d.version)))",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return [tuple(item) for item in json.loads(result.stdout)]


def wait_for(master: int, process: subprocess.Popen[bytes], output: bytearray, marker: str, deadline: float) -> None:
    expected = marker.encode()
    start = len(output)
    while expected not in output[start:]:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"TTY 等待提示超时：{marker}\n{output.decode(errors='replace')}")
        if process.poll() is not None:
            raise RuntimeError(f"TTY 在提示前退出：{marker}\n{output.decode(errors='replace')}")
        readable, _, _ = select.select([master], [], [], 0.1)
        if not readable:
            continue
        try:
            output.extend(os.read(master, 4096))
        except OSError as error:
            if error.errno == errno.EIO:
                raise RuntimeError(f"TTY 在提示前关闭：{marker}") from error
            raise


def verify_tty_flow(binary: Path, python: str, project: Path) -> str:
    if not (project / ".model-maven/project.json").is_file():
        (project / "scaffold_test_model.py").write_text(
            "from torch import nn\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.linear = nn.Linear(4, 2)\n"
            "    def forward(self, values):\n"
            "        return self.linear(values)\n"
            "def build_model():\n"
            "    return Model()\n",
            encoding="utf-8",
        )
        run(binary, python, project, "init", "--model", "scaffold_test_model:build_model")
    extension_store = project / ".model-maven/extensions"
    before_extensions = {path.name for path in extension_store.iterdir()} if extension_store.is_dir() else set()
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(binary), "--python", python],
        cwd=project,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()
    deadline = time.monotonic() + 180
    try:
        wait_for(master, process, output, "选择操作:", deadline)
        os.write(master, "组件\n".encode())
        wait_for(master, process, output, "选择操作:", deadline)
        os.write(master, "浏览组件目录\n".encode())
        wait_for(master, process, output, "搜索组件", deadline)
        os.write(master, b"\n")
        wait_for(master, process, output, "留空返回", deadline)
        os.write(master, b"scaffold\n")
        wait_for(master, process, output, "新组件 ID", deadline)
        os.write(master, b"verify/tty-layer\n")
        wait_for(master, process, output, "类型：", deadline)
        os.write(master, b"layer\n")
        wait_for(master, process, output, "初始版本", deadline)
        os.write(master, b"\n")
        wait_for(master, process, output, "项目内输出目录", deadline)
        os.write(master, b"extensions/tty-layer\n")
        wait_for(master, process, output, "选择操作:", deadline)
        if not (project / "extensions/tty-layer/pyproject.toml").is_file():
            raise RuntimeError("TTY scaffold 菜单没有生成独立包。")
        os.write(master, "浏览组件目录\n".encode())
        wait_for(master, process, output, "搜索组件", deadline)
        os.write(master, b"\n")
        wait_for(master, process, output, "留空返回", deadline)
        os.write(master, b"install\n")
        wait_for(master, process, output, "扩展包目录", deadline)
        os.write(master, b"extensions/tty-layer\n")
        wait_for(master, process, output, "[y/N]", deadline)
        os.write(master, b"y\n")
        wait_for(master, process, output, "选择操作:", deadline)
        variants = project / ".model-maven/variants"
        variants.mkdir(parents=True, exist_ok=True)
        (variants / "tty-reference.json").write_text(
            json.dumps({"edits": [{"action": "add", "component_ref": {"id": "verify/tty-layer", "version": "0.1.0"}}]}),
            encoding="utf-8",
        )
        os.write(master, "浏览组件目录\n".encode())
        wait_for(master, process, output, "搜索组件", deadline)
        os.write(master, b"verify/tty-layer\n")
        wait_for(master, process, output, "留空返回", deadline)
        os.write(master, b"uninstall\n")
        wait_for(master, process, output, "精确组件版本", deadline)
        os.write(master, b"verify/tty-layer@0.1.0\n")
        wait_for(master, process, output, "输入 FORCE 接受可能的重载失败并继续", deadline)
        os.write(master, b"FORCE\n")
        wait_for(master, process, output, "选择操作:", deadline)
        run_id = "run-1700000000000-1-1"
        run_directory = project / ".model-maven/runs" / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "run.json").write_text(
            json.dumps({"schema_version": 1, "id": run_id, "status": "completed", "kept": False}),
            encoding="utf-8",
        )
        (run_directory / "events.jsonl").write_text('{"event":"finished"}\n', encoding="utf-8")
        (run_directory / "worker.log").write_text("tty log\n", encoding="utf-8")
        (run_directory / "progress.jsonl").write_text("tty progress\n", encoding="utf-8")
        os.write(master, "返回首页\n".encode())
        wait_for(master, process, output, "选择操作:", deadline)
        os.write(master, "产物与清理\n".encode())
        wait_for(master, process, output, "选择操作:", deadline)
        os.write(master, "清理运行文件\n".encode())
        wait_for(master, process, output, "运行 ID", deadline)
        os.write(master, b"\n")
        wait_for(master, process, output, "[Y/n]", deadline)
        os.write(master, b"\n")
        wait_for(master, process, output, "预览并释放某个已结束运行的 checkpoint", deadline)
        os.write(master, b"\n")
        wait_for(master, process, output, "输入 CLEAN", deadline)
        os.write(master, b"CLEAN\n")
        wait_for(master, process, output, "选择操作:", deadline)
        if (run_directory / "worker.log").exists() or (run_directory / "progress.jsonl").exists():
            raise RuntimeError("TTY clean 没有移除确认计划中的运行日志与进度文件。")
        if not (run_directory / "run.json").is_file() or "run_artifact_cleaned" not in (run_directory / "events.jsonl").read_text(encoding="utf-8"):
            raise RuntimeError("TTY clean 未保留运行摘要并记录产物清理事件。")
        os.write(master, b"0\n")
        process.wait(timeout=max(1, deadline - time.monotonic()))
        if process.returncode != 0:
            raise RuntimeError(f"TTY 工作台退出码 {process.returncode}。")
        text = output.decode("utf-8", errors="replace")
        if "verify/tty-layer" not in text or "正在按 pyproject.toml 构建扩展 wheel" not in text or "已移除组件 verify/tty-layer@0.1.0" not in text:
            raise RuntimeError(f"TTY scaffold/install/uninstall 流程缺少预期反馈：\n{text}")
        after_extensions = {path.name for path in extension_store.iterdir()} if extension_store.is_dir() else set()
        if after_extensions != before_extensions:
            raise RuntimeError("TTY uninstall 没有仅清理新安装、已确认移除的项目 wheel。")
        if not (variants / "tty-reference.json").is_file():
            raise RuntimeError("强制卸载意外改写了用户的模型变体文件。")
        return "component_catalog_scaffold_install_reference_checked_uninstall_and_run_artifact_clean"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="验收 Layer/Provider scaffold、本地 wheel 安装/卸载、静态发现及 runtime test 的完整流程。"
    )
    parser.add_argument("--binary", type=Path, default=repository / "target/debug/m-model")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        parser.error(f"找不到 CLI 二进制：{binary}；先运行 cargo build。")
    try:
        subprocess.run([args.python, "-c", "import torch"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error(f"所选 Python 必须能导入 PyTorch：{error}")

    before_environment = inventory(args.python)
    with tempfile.TemporaryDirectory(prefix="model-maven-component-scaffold-") as temporary:
        project = Path(temporary)
        expected = [
            ("layer", "verify/gated-block", "gated-block"),
            ("layer-edit-provider", "verify/insert-gate", "insert-gate"),
        ]
        expected_components = ["verify/gated-block", "verify/gated-block-alias", "verify/insert-gate"]
        for kind, component_id, directory in expected:
            output = f"extensions/{directory}"
            scaffold = run(
                binary,
                args.python,
                project,
                "component",
                "scaffold",
                component_id,
                "--kind",
                kind,
                "--output",
                output,
            )
            if scaffold["data"]["manifest"]["kind"] != kind:
                raise RuntimeError(f"{component_id} scaffold 生成了错误 kind。")
            package = project / output
            required = [package / "pyproject.toml", package / "README.md"]
            if kind == "layer":
                required.extend(
                    [
                        package / "src/model_maven_verify_gated_block/layer.py",
                        package / "src/model_maven_verify_gated_block/__model_maven__/gated-block.json",
                    ]
                )
            else:
                required.extend(
                    [
                        package / "src/model_maven_verify_insert_gate/provider.py",
                        package / "src/model_maven_verify_insert_gate/__model_maven__/insert-gate.json",
                    ]
                )
            if any(not path.is_file() for path in required):
                raise RuntimeError(f"{component_id} scaffold 缺少标准包文件。")

            sentinel = package / "README.md"
            sentinel.write_text("preserve existing extension", encoding="utf-8")
            collision = run(
                binary,
                args.python,
                project,
                "component",
                "scaffold",
                component_id,
                "--kind",
                kind,
                "--output",
                output,
                expect_success=False,
            )
            if sentinel.read_text(encoding="utf-8") != "preserve existing extension":
                raise RuntimeError("重复 scaffold 覆盖了现有文件。")
            if collision["diagnostics"][0]["code"] != "COMPONENT_SCAFFOLD_FAILED":
                raise RuntimeError("重复 scaffold 没有返回明确的目录冲突诊断。")

            if component_id == "verify/gated-block":
                entrypoint = '"gated-block" = "model_maven_verify_gated_block.layer:make_layer"'
                pyproject = package / "pyproject.toml"
                pyproject_text = pyproject.read_text(encoding="utf-8")
                if pyproject_text.count(entrypoint) != 1:
                    raise RuntimeError("Layer package scaffold 的 entry point 与预期不符。")
                pyproject.write_text(
                    pyproject_text.replace(
                        entrypoint,
                        entrypoint + '\n"gated-block-alias" = "model_maven_verify_gated_block.layer:make_layer"',
                    ),
                    encoding="utf-8",
                )
                manifest_path = package / "src/model_maven_verify_gated_block/__model_maven__/gated-block.json"
                alias_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                alias_manifest["component_id"] = "verify/gated-block-alias"
                (manifest_path.parent / "gated-block-alias.json").write_text(
                    json.dumps(alias_manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

            installed = run(
                binary,
                args.python,
                project,
                "component",
                "install",
                output,
                "--no-build-isolation",
            )
            installation = installed["data"]["installation"]
            if installation["environment_modified"] is not False:
                raise RuntimeError("组件安装意外修改了所选 Python 环境。")
            package_root = project / installation["site_packages"]
            if not package_root.is_dir() or package_root.parent.name != "extensions":
                raise RuntimeError("wheel 未安装到项目级扩展存储。")
            print(f"installed {component_id}@{installation['distribution_version']}", flush=True)

        discovered = run(binary, args.python, project, "component", "discover")
        if discovered["data"]["found"] != len(expected_components):
            raise RuntimeError("默认 component discover 没有扫描到所有本地安装组件。")

        for component_id in expected_components:
            report = run(binary, args.python, project, "component", "test", component_id)
            if report["data"].get("status") != "passed":
                raise RuntimeError(f"{component_id} runtime conformance check 未通过。")
            print(f"tested {component_id}: {report['data'].get('coverage')}", flush=True)

        tty_flow = verify_tty_flow(binary, args.python, project)

        (project / "use_extension.py").write_text(
            "from model_maven_verify_insert_gate.provider import apply\n",
            encoding="utf-8",
        )
        provider_preview = run(binary, args.python, project, "component", "uninstall", "verify/insert-gate@0.1.0")
        if (
            provider_preview["data"].get("applied") is not False
            or provider_preview["data"].get("wheel_removed") is not True
            or provider_preview["data"].get("requires_force") is not True
            or not provider_preview["data"].get("source_imports")
        ):
            raise RuntimeError("component uninstall 未发现项目源码中的 Provider 导入。")
        (project / "use_extension.py").unlink()
        provider_preview = run(binary, args.python, project, "component", "uninstall", "verify/insert-gate@0.1.0")
        removed_provider = run(binary, args.python, project, "component", "uninstall", "verify/insert-gate@0.1.0", "--apply")
        provider_root = project / provider_preview["data"]["extension_path"]
        if removed_provider["data"].get("applied") is not True or provider_root.exists():
            raise RuntimeError("无引用 Provider 未能从项目注册表和 wheel 存储中移除。")

        variants = project / ".model-maven/variants"
        variants.mkdir(parents=True, exist_ok=True)
        (variants / "uses-gated-block.json").write_text(
            json.dumps({"edits": [{"action": "add", "component_ref": {"id": "verify/gated-block", "version": "0.1.0"}}]}),
            encoding="utf-8",
        )
        layer_preview = run(binary, args.python, project, "component", "uninstall", "verify/gated-block@0.1.0")
        if layer_preview["data"].get("requires_force") is not True or len(layer_preview["data"].get("variant_references", [])) != 1:
            raise RuntimeError("component uninstall 未报告会失效的已保存变体引用。")
        rejected = run(
            binary, args.python, project,
            "component", "uninstall", "verify/gated-block@0.1.0", "--apply",
            expect_success=False,
        )
        if rejected["diagnostics"][0]["code"] != "COMPONENT_IN_USE":
            raise RuntimeError("引用中的组件没有要求显式 force 确认。")
        if not (project / layer_preview["data"]["extension_path"]).is_dir():
            raise RuntimeError("被引用组件在无 force 的失败尝试中被删除。")

        run_dir = project / ".model-maven/runs/run-uninstall-guard"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps({"schema_version": 1, "id": "run-uninstall-guard", "status": "running"}),
            encoding="utf-8",
        )
        active_rejected = run(
            binary, args.python, project,
            "component", "uninstall", "verify/gated-block@0.1.0", "--apply", "--force",
            expect_success=False,
        )
        if active_rejected["diagnostics"][0]["code"] != "COMPONENT_ACTIVE_RUNS":
            raise RuntimeError(
                "存在活动训练任务时未阻止扩展卸载："
                + json.dumps(active_rejected, ensure_ascii=False)
            )
        if not (project / layer_preview["data"]["extension_path"]).is_dir():
            raise RuntimeError("活动运行保护失败后组件 wheel 被删除。")
        import shutil
        shutil.rmtree(run_dir)

        run(binary, args.python, project, "component", "uninstall", "verify/gated-block@0.1.0", "--apply", "--force")
        shared_root = project / layer_preview["data"]["extension_path"]
        if not shared_root.is_dir():
            raise RuntimeError("移除一个组件时误删了仍被同一 wheel 的另一个组件使用的文件。")
        alias_preview = run(binary, args.python, project, "component", "uninstall", "verify/gated-block-alias@0.1.0")
        if alias_preview["data"].get("wheel_removed") is not True:
            raise RuntimeError("最后一个组件卸载时没有计划清理共享 wheel。")
        run(binary, args.python, project, "component", "uninstall", "verify/gated-block-alias@0.1.0", "--apply")
        if shared_root.exists():
            raise RuntimeError("最后一个组件卸载后仍残留共享 wheel。")

        missing_build_directory = project / "extensions/missing-build-dependency"
        missing_build_directory.mkdir(parents=True)
        (missing_build_directory / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["model-maven-build-backend-that-does-not-exist-938271==1.0.0"]\n'
            'build-backend = "missing_backend"\n',
            encoding="utf-8",
        )
        extension_store = project / ".model-maven/extensions"
        extensions_before_build_failure = set(extension_store.iterdir()) if extension_store.is_dir() else set()
        missing_build = run(
            binary,
            args.python,
            project,
            "component",
            "install",
            "extensions/missing-build-dependency",
            "--no-build-isolation",
            expect_success=False,
        )
        if missing_build["diagnostics"][0]["code"] != "COMPONENT_BUILD_REQUIREMENTS_MISSING":
            raise RuntimeError("缺失的本地构建依赖没有在启动 backend 前被快速指出。")
        extensions_after_build_failure = set(extension_store.iterdir()) if extension_store.is_dir() else set()
        if extensions_after_build_failure != extensions_before_build_failure:
            raise RuntimeError("构建依赖预检失败时修改了项目扩展存储。")

        missing_directory = project / "extensions/missing-dependency"
        missing_directory.mkdir(parents=True)
        (missing_directory / "pyproject.toml").write_text(
            "[build-system]\nrequires = [\"setuptools>=68\", \"wheel\"]\n"
            "build-backend = \"setuptools.build_meta\"\n\n"
            "[project]\nname = \"model-maven-missing-dependency-check\"\n"
            "version = \"0.1.0\"\nrequires-python = \">=3.10\"\n"
            "dependencies = [\"model-maven-package-that-does-not-exist-938271==1.0.0\"]\n\n"
            "[project.entry-points.\"model_maven.components\"]\n"
            "missing = \"model_maven_missing.layer:make_layer\"\n\n"
            "[tool.setuptools]\npackage-dir = {\"\" = \"src\"}\n\n"
            "[tool.setuptools.packages.find]\nwhere = [\"src\"]\n\n"
            "[tool.setuptools.package-data]\n"
            "model_maven_missing = [\"__model_maven__/*.json\"]\n",
            encoding="utf-8",
        )
        source = missing_directory / "src/model_maven_missing"
        (source / "__model_maven__").mkdir(parents=True)
        (source / "__init__.py").write_text("", encoding="utf-8")
        (source / "layer.py").write_text(
            "from torch import nn\n\ndef make_layer():\n    return nn.Identity()\n",
            encoding="utf-8",
        )
        (source / "__model_maven__/missing.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "namespace": "verify",
                    "component_id": "verify/missing",
                    "component_version": "0.1.0",
                    "extension_api_version": 1,
                    "kind": "layer",
                    "entrypoint": "model_maven_missing.layer:make_layer",
                    "description": "Dependency preflight fixture.",
                    "params_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                    "capabilities": {"construct": {"state": "supported"}},
                }
            ),
            encoding="utf-8",
        )
        failed = run(
            binary,
            args.python,
            project,
            "component",
            "install",
            "extensions/missing-dependency",
            expect_success=False,
        )
        if failed["diagnostics"][0]["code"] != "COMPONENT_DEPENDENCY_MISSING":
            raise RuntimeError("缺失的 Python 依赖没有在写入本地扩展存储前被拒绝。")

    after_environment = inventory(args.python)
    if before_environment != after_environment:
        raise RuntimeError("组件 scaffold/install/test 改变了所选 Python distribution 清单。")
    print(json.dumps({"status": "passed", "components": 3, "tty_flow": tty_flow, "uninstall_reference_checks": "passed", "shared_wheel_retention": "passed", "python_environment_unchanged": True}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"component scaffold acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
