#!/usr/bin/env python3
"""Exercise user-owned training loops and human TTY model/Layer workflows."""

from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def invoke(binary: Path, python: str, cwd: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    command = [str(binary), "--python", python, "--json", *args]
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"CLI 未返回 JSON 报告：{' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from error
    return result, report


def invoke_success(binary: Path, python: str, cwd: Path, *args: str) -> dict:
    result, report = invoke(binary, python, cwd, *args)
    if result.returncode != 0 or report.get("status") != "success":
        raise RuntimeError(
            f"CLI 操作失败：{' '.join(args)}\n"
            f"{json.dumps(report, ensure_ascii=False, indent=2)}\nstderr:\n{result.stderr}"
        )
    return report


def verify_tty_loop_flow(binary: Path, python: str, cwd: Path) -> str:
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(binary), "--python", python],
        cwd=cwd,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()
    deadline = time.monotonic() + 45
    try:
        # The workbench now renders category-local prompts as “选择操作:”.
        # Older acceptance flows waited for the retired one-word prompt.
        while "选择操作:".encode() not in output:
            if time.monotonic() > deadline or process.poll() is not None:
                raise RuntimeError("TTY 工作台没有进入主菜单。")
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    output.extend(os.read(master, 4096))
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    raise RuntimeError("TTY 工作台在菜单前退出。") from error

        # Let a buggy generic batch prompt consume the first zero and still
        # exit the menu after reporting its diagnostic.
        os.write(
            master,
            (
                "3\n"  # Home -> try/train category.
                "2\n"  # Category-local action: try training steps.
                "3\n"  # User-owned external loop.
                "external_loop_fixture:run_loop\n"
                "2\n"
                "lr\n"
                "0.05\n"
                "\n"
                "tty-loop\n"
                "start\n"
                "back\n"
                "8\n"
            ).encode(),
        )
        while process.poll() is None:
            if time.monotonic() > deadline:
                process.kill()
                raise RuntimeError(
                    "TTY 完整循环试跑超过 45 秒；最近终端输出：\n"
                    + output.decode("utf-8", errors="replace")[-6000:]
                )
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    output.extend(os.read(master, 4096))
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
        process.wait(timeout=2)
        while select.select([master], [], [], 0)[0]:
            try:
                output.extend(os.read(master, 4096))
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
        rendered = output.decode("utf-8", errors="replace")
        if (
            process.returncode != 0
            or "完整自有循环试跑：2/2 步" not in rendered
            or "EXTERNAL_LOOP_BATCH_UNUSED" in rendered
            or "示例批次入口" in rendered
        ):
            raise RuntimeError(f"TTY 完整循环交互不符合预期：\n{rendered}")
        return "menu_to_loop_trial_without_batch_prompt"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


def verify_tty_layer_browser(binary: Path, python: str, cwd: Path) -> str:
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(binary), "--python", python],
        cwd=cwd,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()
    deadline = time.monotonic() + 45
    try:
        while "选择操作:".encode() not in output:
            if time.monotonic() > deadline or process.poll() is not None:
                raise RuntimeError("TTY Layer 浏览没有进入主菜单。")
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    output.extend(os.read(master, 4096))
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    raise RuntimeError("TTY Layer 浏览在菜单前退出。") from error

        os.write(
            master,
            (
                "1\n"
                "3\n"
                "Linear\n"
                "4\n"
                "\n"
                "back\n"
                "back\n"
                "8\n"
            ).encode(),
        )
        while process.poll() is None:
            if time.monotonic() > deadline:
                process.kill()
                raise RuntimeError("TTY Layer 搜索/查看超过 45 秒。")
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    output.extend(os.read(master, 4096))
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
        process.wait(timeout=2)
        while select.select([master], [], [], 0)[0]:
            try:
                output.extend(os.read(master, 4096))
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
        rendered = output.decode("utf-8", errors="replace")
        if (
            process.returncode != 0
            or "model.layers.1" not in rendered
            or not any(label in rendered for label in ("可改参数", "变体可改字段"))
            or "Linear" not in rendered
            or not any(label in rendered for label in ("创建位置", "类型定义"))
        ):
            raise RuntimeError(f"TTY Layer 搜索/查看交互不符合预期：\n{rendered}")
        return "menu_to_layer_type_search_and_detail"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


def verify_tty_grouped_structure_plan(binary: Path, python: str, cwd: Path) -> str:
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(binary), "--python", python],
        cwd=cwd,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()
    deadline = time.monotonic() + 60
    try:
        while "选择操作:".encode() not in output:
            if time.monotonic() > deadline or process.poll() is not None:
                raise RuntimeError("TTY 成组结构编辑没有进入主菜单。")
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    output.extend(os.read(master, 4096))
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    raise RuntimeError("TTY 成组结构编辑在菜单前退出。") from error

        os.write(
            master,
            (
                "2\n"
                "12\n"
                "add\n"
                "model.blocks\n"
                "before\n"
                "1\n"
                "Dropout\n"
                "\n"
                "\n"
                "yes\n"
                "remove\n"
                "model.blocks.3\n"
                "\n"
                "yes\n"
                "add\n"
                "model.branches\n"
                "before\n"
                "right\n"
                "projection\n"
                "Identity\n"
                "yes\n"
                "remove\n"
                "model.branches.projection\n"
                "\n"
                "\n"
                "yes\n"
                "back\n"
                "yes\n"
                "16\n"
                "8\n"
            ).encode(),
        )
        while process.poll() is None:
            if time.monotonic() > deadline:
                process.kill()
                raise RuntimeError(
                    "TTY 成组结构计划超过 60 秒；最近终端输出：\n"
                    + output.decode("utf-8", errors="replace")[-6000:]
                )
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    output.extend(os.read(master, 4096))
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
        process.wait(timeout=2)
        while select.select([master], [], [], 0)[0]:
            try:
                output.extend(os.read(master, 4096))
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
        rendered = output.decode("utf-8", errors="replace")
        source_path = cwd / "external_loop_fixture.py"
        source = source_path.read_text(encoding="utf-8")
        if (
            process.returncode != 0
            or rendered.count("已刷新计划模型树") != 4
            or "确认按以上完整结构计划写入源码" not in rendered
            or "# Rebuild this ModuleList" not in source
            or "self.blocks[1]," not in source
        ):
            raise RuntimeError(
                "TTY 成组结构计划没有按候选结构刷新模型树：\n"
                f"{rendered}\n生成源码：\n{source}"
            )
        return "source_plan_refreshes_tree_after_each_structural_edit"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="验收完整自有训练循环、状态协议和 TTY Layer 搜索选择流程。"
    )
    parser.add_argument("--binary", type=Path, default=repository / "target/debug/m-model")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        parser.error(f"找不到 CLI 二进制：{binary}；先运行 cargo build。")

    with tempfile.TemporaryDirectory(prefix="model-maven-external-loop-") as temporary:
        project = Path(temporary)
        (project / "external_loop_fixture.py").write_text(
            "import time\n"
            "import torch\n"
            "from torch import nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.projection = nn.Linear(4, 2)\n\n"
            "    def forward(self, values):\n"
            "        return self.projection(values)\n\n"
            "def build_model():\n"
            "    return Model()\n\n"
            "class BrowserModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.layers = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))\n\n"
            "    def forward(self, values):\n"
            "        return self.layers(values)\n\n"
            "def build_browser_model():\n"
            "    return BrowserModel()\n\n"
            "class ModuleListModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.ReLU(), nn.Identity(), nn.Linear(4, 4)])\n\n"
            "        self.branches = nn.ModuleDict({'left': nn.Identity(), 'right': nn.Identity()})\n\n"
            "    def forward(self, values):\n"
            "        for block in self.blocks:\n"
            "            values = block(values)\n"
            "        return values\n\n"
            "def build_modulelist_model():\n"
            "    return ModuleListModel()\n\n"
            "def modulelist_batch():\n"
            "    return torch.randn(2, 4)\n\n"
            "def run_loop(model, context, lr=0.05, pause=0.0):\n"
            "    optimizer = getattr(run_loop, '_optimizer', None)\n"
            "    if optimizer is None:\n"
            "        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)\n"
            "        pending = getattr(run_loop, '_pending_state', None)\n"
            "        if pending is not None:\n"
            "            optimizer.load_state_dict(pending)\n"
            "        run_loop._optimizer = optimizer\n"
            "    for _ in range(context.remaining_steps):\n"
            "        if context.stop_requested:\n"
            "            break\n"
            "        step = context.completed_steps + 1\n"
            "        values = torch.full((3, 4), float(step))\n"
            "        target = torch.zeros((3, 2))\n"
            "        optimizer.zero_grad(set_to_none=True)\n"
            "        loss = (model(values) - target).square().mean()\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "        if pause:\n"
            "            time.sleep(pause)\n"
            "        if not context.report_step(loss.detach(), {'synthetic_step': step}):\n"
            "            break\n"
            "    return {'loop': 'fixture'}\n\n"
            "def _save_loop_state():\n"
            "    return run_loop._optimizer.state_dict()\n\n"
            "def _load_loop_state(state):\n"
            "    run_loop._pending_state = state\n\n"
            "run_loop.state_dict = _save_loop_state\n"
            "run_loop.load_state_dict = _load_loop_state\n\n"
            "def run_without_state(model, context):\n"
            "    for _ in range(context.remaining_steps):\n"
            "        if context.stop_requested:\n"
            "            break\n"
            "        with torch.no_grad():\n"
            "            for parameter in model.parameters():\n"
            "                parameter.add_(0.001)\n"
            "        if not context.report_step(0.0, {'stateful': 0}):\n"
            "            break\n"
            "def run_stateless(model, context):\n"
            "    return run_without_state(model, context)\n"
            "run_stateless.MODEL_MAVEN_STATELESS = True\n",
            encoding="utf-8",
        )

        # The human TTY flows intentionally start from the project workbench.
        # Persist the same model entry that the earlier JSON-only flows use;
        # without this, the current first-run onboarding screen is correctly
        # shown and its choices consume the obsolete global menu sequence.
        invoke_success(
            binary,
            args.python,
            project,
            "init",
            "--model",
            "external_loop_fixture:build_model",
        )

        trial = invoke_success(
            binary,
            args.python,
            project,
            "try",
            "--model",
            "external_loop_fixture:build_model",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop_fixture:run_loop",
            "--loop-param",
            "lr=0.05",
            "--steps",
            "2",
        )
        if (
            trial["data"].get("completed_steps") != 2
            or trial["data"].get("capabilities", {}).get("backward_and_optimizer_updates")
            != "not_independently_verified"
        ):
            raise RuntimeError("external-loop 无批次有限试跑未按协议报告步骤及能力边界。")

        first = invoke_success(
            binary,
            args.python,
            project,
            "train",
            "--model",
            "external_loop_fixture:build_model",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop_fixture:run_loop",
            "--loop-param",
            "lr=0.05",
            "--steps",
            "3",
            "--checkpoint-every",
            "2",
        )
        first_data = first["data"]
        if (
            first_data.get("training_mode") != "external-loop"
            or first_data.get("completed_steps") != 3
            or first_data.get("checkpoint_state", {}).get("external_loop") != "saved"
            or first_data.get("capabilities", {}).get("backward_and_optimizer_updates")
            != "not_independently_verified"
            or first_data.get("mean_metrics", {}).get("synthetic_step") != 2.0
        ):
            raise RuntimeError(
                "external-loop 报告、状态保存或未验证能力边界不正确：\n"
                f"{json.dumps(first_data, ensure_ascii=False, indent=2)}"
            )

        resumed = invoke_success(
            binary,
            args.python,
            project,
            "train",
            "--resume",
            str(first_data["run_id"]),
            "--additional-steps",
            "2",
            "--checkpoint-every",
            "2",
        )
        resumed_data = resumed["data"]
        if (
            resumed_data.get("completed_steps") != 5
            or resumed_data.get("resume_state", {}).get("external_loop") != "restored"
            or resumed_data.get("checkpoint_state", {}).get("external_loop") != "saved"
        ):
            raise RuntimeError(
                "external-loop 自定义 optimizer 状态没有从 checkpoint 恢复：\n"
                f"{json.dumps(resumed_data, ensure_ascii=False, indent=2)}"
            )

        stateless = invoke_success(
            binary,
            args.python,
            project,
            "train",
            "--model",
            "external_loop_fixture:build_model",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop_fixture:run_without_state",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        )
        stateless_id = stateless["data"]["run_id"]
        missing_state, missing_report = invoke(
            binary,
            args.python,
            project,
            "train",
            "--resume",
            str(stateless_id),
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
        )
        if (
            missing_state.returncode == 0
            or missing_report.get("diagnostics", [{}])[0].get("code") != "RESUME_STATE_UNAVAILABLE"
        ):
            raise RuntimeError("没有状态钩子的循环被静默续训，而不是要求显式选择状态重置。")

        reset = invoke_success(
            binary,
            args.python,
            project,
            "train",
            "--resume",
            str(stateless_id),
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
            "--allow-state-reset",
        )
        if reset["data"].get("resume_state", {}).get("external_loop") != "reset":
            raise RuntimeError("显式 --allow-state-reset 没有记录循环状态已重置。")

        stateless = invoke_success(
            binary,
            args.python,
            project,
            "train",
            "--model",
            "external_loop_fixture:build_model",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop_fixture:run_stateless",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        )
        if stateless["data"].get("checkpoint_state", {}).get("external_loop") != "stateless":
            raise RuntimeError("显式 MODEL_MAVEN_STATELESS 声明的 checkpoint 状态不准确。")
        stateless_resume = invoke_success(
            binary,
            args.python,
            project,
            "train",
            "--resume",
            str(stateless["data"]["run_id"]),
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
        )
        if stateless_resume["data"].get("resume_state", {}).get("external_loop") != "stateless":
            raise RuntimeError("显式无状态循环恢复没有准确记录其状态协议。")

        packaged_loop = project / "packaged_loop.py"
        shutil.copyfile(repository / "examples/external_loop.py", packaged_loop)
        packaged = invoke_success(
            binary,
            args.python,
            project,
            "train",
            "--model",
            "packaged_loop:build_model",
            "--training-mode",
            "external-loop",
            "--loop",
            "packaged_loop:run_loop",
            "--loop-param",
            "lr=0.01",
            "--steps",
            "3",
            "--checkpoint-every",
            "1",
        )
        packaged_data = packaged["data"]
        if packaged_data.get("loop_result", {}).get("data_cursor") != 3:
            raise RuntimeError("仓库完整循环样例没有返回预期的数据游标。")
        packaged_resume = invoke_success(
            binary,
            args.python,
            project,
            "train",
            "--resume",
            str(packaged_data["run_id"]),
            "--additional-steps",
            "2",
            "--checkpoint-every",
            "1",
        )
        packaged_resume_data = packaged_resume["data"]
        if (
            packaged_resume_data.get("loop_result", {}).get("data_cursor") != 5
            or packaged_resume_data.get("resume_state", {}).get("external_loop") != "restored"
        ):
            raise RuntimeError("仓库完整循环样例的 optimizer 与数据游标没有随 checkpoint 恢复。")

        tty_flow = "skipped_non_linux"
        tty_layer_browser = "skipped_non_linux"
        if sys.platform.startswith("linux"):
            tty_flow = verify_tty_loop_flow(binary, args.python, project)
            browser_project = project / "tty-layer-browser"
            browser_project.mkdir()
            shutil.copyfile(project / "external_loop_fixture.py", browser_project / "external_loop_fixture.py")
            invoke_success(
                binary,
                args.python,
                browser_project,
                "init",
                "--model",
                "external_loop_fixture:build_browser_model",
            )
            tty_layer_browser = verify_tty_layer_browser(binary, args.python, browser_project)

            structure_project = project / "tty-structure-plan"
            structure_project.mkdir()
            shutil.copyfile(project / "external_loop_fixture.py", structure_project / "external_loop_fixture.py")
            invoke_success(
                binary,
                args.python,
                structure_project,
                "init",
                "--model",
                "external_loop_fixture:build_modulelist_model",
                "--batch",
                "external_loop_fixture:modulelist_batch",
            )
            tty_structure_plan = verify_tty_grouped_structure_plan(
                binary, args.python, structure_project
            )

        cooperative_stop = "skipped_non_linux"
        if sys.platform.startswith("linux"):
            detached = invoke_success(
                binary,
                args.python,
                project,
                "train",
                "--model",
                "external_loop_fixture:build_model",
                "--training-mode",
                "external-loop",
                "--loop",
                "external_loop_fixture:run_loop",
                "--loop-param",
                "pause=0.01",
                "--steps",
                "10000",
                "--checkpoint-every",
                "1000",
                "--detach",
            )
            detached_id = detached["data"]["run_id"]
            for _ in range(200):
                attached = invoke_success(
                    binary,
                    args.python,
                    project,
                    "runs",
                    "attach",
                    detached_id,
                    "--follow=false",
                )
                progress = attached["data"].get("progress", [])
                if progress and progress[-1].get("step", 0) >= 1:
                    break
                time.sleep(0.025)
            else:
                raise RuntimeError("后台 external-loop 没有报告第一个步骤。")

            invoke_success(binary, args.python, project, "runs", "stop", detached_id)
            for _ in range(200):
                stopped = invoke_success(binary, args.python, project, "runs", "show", detached_id)
                record = stopped["data"].get("record", {})
                if record.get("status") in {"interrupted", "completed", "failed"}:
                    break
                time.sleep(0.025)
            if record.get("status") != "interrupted" or stopped["data"].get("checkpoint", {}).get("completed_steps", 0) < 1:
                raise RuntimeError(
                    "external-loop 未在协作停止边界写入恢复点并标记中断："
                    f"{json.dumps(stopped['data'], ensure_ascii=False, indent=2)}"
                )
            cooperative_stop = "stopped_at_reported_checkpoint_boundary"

        print(json.dumps({
            "status": "passed",
            "external_loop_steps": first_data["completed_steps"],
            "external_loop_trial_steps": trial["data"]["completed_steps"],
            "stateful_resume_steps": resumed_data["completed_steps"],
            "optimizer_state_hook": "restored",
            "missing_state_resume": "rejected_until_explicit_reset",
            "explicit_state_reset": "recorded",
            "stateless_checkpoint_resume": "declared_and_restored",
            "shipped_callable_example": "trained_and_resumed_with_optimizer_and_cursor_state",
            "cooperative_stop": cooperative_stop,
            "interactive_tty_flow": tty_flow,
            "interactive_layer_browser": tty_layer_browser,
            "interactive_structure_plan": tty_structure_plan,
            "gradient_and_optimizer_verification": "honestly_reported_unverified",
            "data_factory_required": False,
            "limits": [
                "用户循环负责正确调用 forward/backward/optimizer 和按更新边界报告进度。",
                "步骤预算在 report_step 处检查；长于一次报告间隔的操作由 --timeout 限制，不是 OS 沙箱。",
            ],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"External-loop acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
