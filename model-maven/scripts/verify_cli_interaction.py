#!/usr/bin/env python3
"""Run real-binary JSON and TTY smoke journeys; preserve a diagnostic report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import fcntl
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time


def assertion(step: dict, name: str, passed: bool, message: str) -> None:
    step.setdefault("assertions", []).append(
        {"name": name, "passed": passed, "message": message}
    )
    if not passed:
        step["status"] = "failed"
        step.setdefault("diagnostics", []).append(message)


def validate_matrix(repository: Path) -> dict:
    matrix = json.loads((repository / "docs/cli-acceptance-matrix.json").read_text())
    requirements = json.loads((repository / "docs/acceptance/requirements.json").read_text())
    items = requirements["items"]

    def gates_match(matrix_key: str, requirement_kind: str) -> bool:
        expected = {
            item["source_path"]: (item["requirement"], item["status"])
            for item in items
            if item["kind"] == requirement_kind
        }
        gates = matrix[matrix_key]
        actual = {
            gate["source_path"]: (gate["requirement"], gate["implementation_status_snapshot"])
            for gate in gates
        }
        return len(actual) == len(gates) and actual == expected and all(
            gate["source_path"].endswith(f"/{gate['id']}") and gate.get("required_evidence")
            for gate in gates
        )

    # The matrix records the retired CLI task IDs used by these smoke journeys.
    # Current product work is tracked separately in todo.json.
    cli_tasks = matrix["cli_tasks"]
    archived_cli_tasks = matrix.get("archived_cli_tasks", [])
    cli_task_ids = {task["id"] for task in cli_tasks}
    archived_cli_task_ids = {task["id"] for task in archived_cli_tasks}
    task_rows_valid = (
        len(cli_task_ids) == len(cli_tasks)
        and len(archived_cli_task_ids) == len(archived_cli_tasks)
        and not (archived_cli_task_ids & cli_task_ids)
        and all(task.get("status_snapshot") == "done" for task in archived_cli_tasks)
        and all(
            (repository / task["handoff_file"]).is_file()
            for task in [*cli_tasks, *archived_cli_tasks]
        )
    )
    scenario_ids = {item["id"] for item in items if item["kind"] == "prototype_scenario"}
    gate_ids = {
        gate["id"] for gate in [*matrix["framework_gates"], *matrix["release_gates"]]
    }
    workflow_refs_valid = all(
        set(workflow.get("scenario_refs", [])).issubset(scenario_ids)
        and set(workflow.get("gate_refs", [])).issubset(gate_ids)
        and set(workflow.get("cli_task_refs", [])).issubset(
            cli_task_ids | archived_cli_task_ids
        )
        for workflow in matrix["human_workflows"]
    )
    evidence_layers_explicit = all(
        workflow.get("fixture_type")
        and workflow.get("ui_evidence")
        and workflow.get("service_evidence")
        and workflow.get("external_evidence_required")
        for workflow in matrix["human_workflows"]
    )
    checks = {
        "legacy_revision_matches": matrix["product_revision"] == requirements["document_version"],
        "framework_gate_snapshot_matches": gates_match("framework_gates", "core_validation_gates"),
        "release_gate_snapshot_matches": gates_match("release_gates", "release_gates"),
        "legacy_task_rows_and_handoffs_valid": task_rows_valid,
        "human_workflow_references_valid": workflow_refs_valid,
        "ui_service_external_evidence_split": evidence_layers_explicit,
        "release_gates_stay_open": all(
            gate["release_verification_status"] == "open"
            for gate in matrix["release_gates"]
        ),
    }
    return {
        "id": "acceptance-matrix-integrity",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "diagnostics": [name for name, passed in checks.items() if not passed],
    }


def run_json(binary: Path, python: str, project: Path, command: list[str], timeout: float) -> dict:
    argv = [str(binary), "--python", python, "--json", *command]
    result = subprocess.run(
        argv,
        cwd=project,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    try:
        document = json.loads(result.stdout)
        parse_error = None
    except json.JSONDecodeError as error:
        document = None
        parse_error = str(error)
    return {
        "command": argv,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "report": document,
        "parse_error": parse_error,
    }


def run_tty(binary: Path, python: str, project: Path, timeout: float) -> dict:
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(project)
        environment = os.environ.copy()
        environment.update({"TERM": "xterm-256color", "COLUMNS": "100", "NO_COLOR": "1"})
        argv = [str(binary), "--python", python]
        os.execvpe(str(binary), argv, environment)

    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    input_bytes = (
        b"1\n"  # Choose the named action, not a fixed workbench menu index.
        b"cli_acceptance_model:build_model\n"
        b"cli_acceptance_model:example_batch\n"
        b"1\n"  # Temporary inspection; do not save project defaults.
        b"2\n"  # Probe a real batch.
        b"back\n"
        b"q\n"
    )
    try:
        os.write(master, input_bytes)
    except OSError:
        pass

    transcript = bytearray()
    deadline = time.monotonic() + timeout
    timed_out = False
    wait_status = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            os.kill(pid, signal.SIGKILL)
            break
        ready, _, _ = select.select([master], [], [], min(remaining, 0.25))
        if not ready:
            finished, observed_status = os.waitpid(pid, os.WNOHANG)
            if finished:
                wait_status = observed_status
                pid = -1
                break
            continue
        try:
            chunk = os.read(master, 8192)
        except OSError:
            break
        if not chunk:
            break
        transcript.extend(chunk)

    if pid != -1:
        _, wait_status = os.waitpid(pid, 0)
    if wait_status is None:
        wait_status = 0
    os.close(master)
    exit_code = os.waitstatus_to_exitcode(wait_status)
    return {
        "mode": "tty",
        "exit_code": exit_code,
        "timed_out": timed_out,
        "transcript": transcript.decode("utf-8", errors="replace"),
    }


def write_report(output_dir: Path, report: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    transcript_path = output_dir / "transcript.txt"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    chunks = []
    for step in report.get("steps", []):
        chunks.append(f"## {step.get('id', 'step')} [{step.get('status', 'unknown')}]\n")
        if step.get("transcript"):
            chunks.append(step["transcript"])
        if step.get("stdout"):
            chunks.append("\n--- stdout ---\n" + step["stdout"])
        if step.get("stderr"):
            chunks.append("\n--- stderr ---\n" + step["stderr"])
        for diagnostic in step.get("diagnostics", []):
            chunks.append("\nDIAGNOSTIC: " + diagnostic + "\n")
    transcript_path.write_text("\n".join(chunks))
    return report_path, transcript_path


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="验证 Model-Maven JSON 操作和 TTY 接入流程，并保存逐步诊断。"
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=repository / "target/debug/m-model",
        help="要验收的 m-model 二进制",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="已安装 PyTorch 的 Python 解释器",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "target/cli-acceptance",
        help="保存 report.json 和 transcript.txt 的目录",
    )
    parser.add_argument("--timeout", type=float, default=45.0, help="每个操作的超时秒数")
    parser.add_argument(
        "--inject-failure",
        action="store_true",
        help="在所有真实步骤完成后注入一个失败断言，验证失败报告和非零退出",
    )
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report = {
        "schema_version": 1,
        "suite": "cli-interaction-smoke",
        "mode": "real-binary",
        "binary": str(binary),
        "python": args.python,
        "steps": [],
    }

    try:
        if not binary.is_file():
            raise RuntimeError(f"找不到 CLI 二进制：{binary}；先运行 cargo build --locked。")
        if args.timeout <= 0:
            raise RuntimeError("--timeout 必须大于 0。")
        matrix_step = validate_matrix(repository)
        report["steps"].append(matrix_step)
        if matrix_step["status"] != "passed":
            raise RuntimeError(
                "CLI acceptance matrix and historical requirement records are inconsistent: "
                + ", ".join(matrix_step["diagnostics"])
            )
        torch_check = subprocess.run(
            [args.python, "-c", "import torch"], text=True, capture_output=True, timeout=15
        )
        if torch_check.returncode != 0:
            raise RuntimeError(
                f"Python 环境不能导入 PyTorch：{args.python}\n{torch_check.stderr.strip()}"
            )

        with tempfile.TemporaryDirectory(prefix="model-maven-cli-acceptance-") as temporary:
            project = Path(temporary)
            (project / "cli_acceptance_model.py").write_text(
                """import torch
from torch import nn

def build_model():
    return nn.Linear(2, 2)

def example_batch():
    return torch.ones(1, 2)
"""
            )

            inspect = run_json(
                binary,
                args.python,
                project,
                ["model", "inspect", "--model", "cli_acceptance_model:build_model"],
                args.timeout,
            )
            inspect_step = {"id": "json-model-inspect", **inspect, "status": "passed"}
            assertion(
                inspect_step,
                "json_report",
                inspect["exit_code"] == 0
                and inspect["report"] is not None
                and inspect["report"].get("status") == "success",
                "model inspect 应返回退出码 0 和 status=success 的 JSON 报告。",
            )
            if inspect.get("parse_error"):
                inspect_step["diagnostics"].append(
                    f"JSON 解析失败：{inspect['parse_error']}"
                )
            report["steps"].append(inspect_step)

            probe = run_json(
                binary,
                args.python,
                project,
                [
                    "model",
                    "probe",
                    "--model",
                    "cli_acceptance_model:build_model",
                    "--batch",
                    "cli_acceptance_model:example_batch",
                ],
                args.timeout,
            )
            probe_step = {"id": "json-forward-probe", **probe, "status": "passed"}
            calls = (
                probe.get("report", {}).get("data", {}).get("calls", [])
                if isinstance(probe.get("report"), dict)
                else []
            )
            assertion(
                probe_step,
                "real_forward_evidence",
                probe["exit_code"] == 0
                and probe["report"] is not None
                and probe["report"].get("status") == "success"
                and bool(calls),
                "probe 应返回 JSON 成功报告和至少一条真实模块调用证据。",
            )
            if probe.get("parse_error"):
                probe_step["diagnostics"].append(f"JSON 解析失败：{probe['parse_error']}")
            report["steps"].append(probe_step)

            tty = run_tty(binary, args.python, project, args.timeout)
            tty_step = {"id": "tty-existing-model-onboarding", **tty, "status": "passed"}
            transcript = tty["transcript"]
            for name, expected, message in [
                (
                    "human_readable_entry",
                    "接入已有 Python 模型" in transcript,
                    "TTY 应显示已有 Python 模型接入入口。",
                ),
                (
                    "probe_completed",
                    "前向证据" in transcript,
                    "TTY 接入流程应显示模型前向证据。",
                ),
                (
                    "workbench_returned",
                    "模型工作台" in transcript,
                    "探测完成后应回到模型工作台。",
                ),
                (
                    "no_training_prerequisites",
                    "损失函数入口" not in transcript,
                    "仅查看模型和批次不应要求损失函数入口。",
                ),
            ]:
                assertion(tty_step, name, expected, message)
            assertion(
                tty_step,
                "tty_process_exit",
                tty["exit_code"] == 0 and not tty["timed_out"],
                "TTY 流程应在预算内以退出码 0 结束。",
            )
            report["steps"].append(tty_step)

        if args.inject_failure:
            failed_step = {
                "id": "injected-failure",
                "status": "failed",
                "diagnostics": ["CLI-19 验收器故障注入：这是预期的失败诊断。"],
                "assertions": [
                    {
                        "name": "injected_failure",
                        "passed": False,
                        "message": "故障注入应使验收进程非零退出。",
                    }
                ],
            }
            report["steps"].append(failed_step)

    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        report["steps"].append(
            {
                "id": "preflight-or-runner",
                "status": "failed",
                "diagnostics": [str(error)],
            }
        )

    failed = any(step.get("status") == "failed" for step in report["steps"])
    report["status"] = "failed" if failed else "passed"
    report["summary"] = {
        "passed_steps": sum(step.get("status") == "passed" for step in report["steps"]),
        "failed_steps": sum(step.get("status") == "failed" for step in report["steps"]),
        "evidence_boundary": "自动 fixture 证明这些命令和提示在当前环境可运行；不证明真实用户可理解、产品完整或 D0/D1 发布验收通过。",
    }
    report_path, transcript_path = write_report(output_dir, report)
    print(f"CLI 交互验收：{report['status']}")
    print(f"报告：{report_path}")
    print(f"终端记录：{transcript_path}")
    if failed:
        for step in report["steps"]:
            for diagnostic in step.get("diagnostics", []):
                print(f"{step.get('id')}: {diagnostic}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
