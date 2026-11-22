#!/usr/bin/env python3
"""Verify the real workbench cleanup preview can be cancelled without mutations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import tempfile
import time


def run(binary: Path, python: str, timeout: float) -> tuple[dict, str]:
    with tempfile.TemporaryDirectory(prefix="model-maven-cli13-") as directory:
        project = Path(directory)
        initialized = subprocess.run(
            [str(binary), "init", "--model", "fixture:build_model", "--batch", "fixture:example_batch"],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        if initialized.returncode != 0:
            raise RuntimeError(
                f"temporary project initialization failed: {initialized.stdout}\n{initialized.stderr}"
            )

        run_id = "run-100-1-1"
        run_dir = project / ".model-maven" / "runs" / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        files = {
            run_dir / "run.json": json.dumps(
                {"schema_version": 1, "id": run_id, "status": "completed", "kept": False}
            ),
            run_dir / "events.jsonl": '{"event":"finished"}\n',
            run_dir / "worker.log": "worker evidence\n",
            run_dir / "progress.jsonl": "progress evidence\n",
        }
        for path, contents in files.items():
            path.write_text(contents, encoding="utf-8")
        before = {str(path): path.read_bytes() for path in files}

        child_pid, master_fd = pty.fork()
        if child_pid == 0:
            os.chdir(project)
            environment = os.environ.copy()
            environment.update({"TERM": "xterm-256color", "COLUMNS": "100", "NO_COLOR": "1"})
            os.execve(str(binary), [str(binary), "--python", python], environment)

        # Open the outputs/cleanup category, choose cleanup, retain default log/progress
        # selection, reject the CLEAN confirmation, then return home and quit.
        os.write(master_fd, b"6\n2\n\n\n\nnot-clean\n3\n8\n")
        transcript = bytearray()
        deadline = time.monotonic() + timeout
        wait_status = None
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master_fd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                transcript.extend(chunk)
            finished, status = os.waitpid(child_pid, os.WNOHANG)
            if finished:
                wait_status = status
                break
        if wait_status is None:
            os.kill(child_pid, signal.SIGKILL)
            _, wait_status = os.waitpid(child_pid, 0)
        os.close(master_fd)

        output = transcript.decode("utf-8", errors="replace")
        after = {str(path): path.read_bytes() for path in files if path.exists()}
        checks = {
            "workbench_exited_successfully": os.waitstatus_to_exitcode(wait_status) == 0,
            "clean_plan_rendered": "计划 ID：clean-" in output
            and "worker.log" in output
            and "progress.jsonl" in output,
            "explicit_confirmation_requested": "输入 CLEAN 确认固定清单" in output,
            "cancel_result_rendered": "已取消；文件未改变" in output,
            "run_and_files_unchanged": before == after,
        }
        return {
            "status": "verified" if all(checks.values()) else "failed",
            "requirement_id": "CLI13-clean-cancel",
            "command": "m-model init; PTY workbench -> outputs/cleanup -> cancel fixed plan",
            "exit_code": os.waitstatus_to_exitcode(wait_status),
            "checks": checks,
            "fixture": "temporary initialized project with a completed run and two cleanup candidates",
            "environment": {
                "os": sys.platform,
                "python": python,
                "binary": str(binary),
            },
            "scope_limits": [
                "Verifies the real Linux PTY cancellation path only.",
                "Does not verify environment installation or export conflicts.",
            ],
        }, output


def verify_status_change(binary: Path, python: str, timeout: float) -> tuple[dict, str]:
    with tempfile.TemporaryDirectory(prefix="model-maven-cli13-stale-") as directory:
        project = Path(directory)
        initialized = subprocess.run(
            [str(binary), "init", "--model", "fixture:build_model", "--batch", "fixture:example_batch"],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        if initialized.returncode != 0:
            raise RuntimeError(
                f"temporary project initialization failed: {initialized.stdout}\n{initialized.stderr}"
            )
        run_id = "run-101-1-1"
        run_dir = project / ".model-maven" / "runs" / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        run_metadata = run_dir / "run.json"
        run_metadata.write_text(
            json.dumps({"schema_version": 1, "id": run_id, "status": "completed", "kept": False}),
            encoding="utf-8",
        )
        (run_dir / "events.jsonl").write_text('{"event":"finished"}\n', encoding="utf-8")
        worker_log = run_dir / "worker.log"
        progress = run_dir / "progress.jsonl"
        worker_log.write_text("worker evidence\n", encoding="utf-8")
        progress.write_text("progress evidence\n", encoding="utf-8")
        original_log = worker_log.read_bytes()
        original_progress = progress.read_bytes()

        child_pid, master_fd = pty.fork()
        if child_pid == 0:
            os.chdir(project)
            environment = os.environ.copy()
            environment.update({"TERM": "xterm-256color", "COLUMNS": "100", "NO_COLOR": "1"})
            os.execve(str(binary), [str(binary), "--python", python], environment)

        os.write(master_fd, b"6\n2\n\n\n\n")
        transcript = bytearray()
        deadline = time.monotonic() + timeout
        wait_status = None
        confirmation_seen = False
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master_fd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                transcript.extend(chunk)
                if "输入 CLEAN 确认固定清单".encode() in transcript:
                    confirmation_seen = True
                    break
            finished, status = os.waitpid(child_pid, os.WNOHANG)
            if finished:
                wait_status = status
                break

        if confirmation_seen:
            changed_metadata = json.loads(run_metadata.read_text(encoding="utf-8"))
            changed_metadata["status"] = "running"
            run_metadata.write_text(json.dumps(changed_metadata), encoding="utf-8")
            os.write(master_fd, b"CLEAN\n3\n8\n")

        while wait_status is None and time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master_fd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                transcript.extend(chunk)
            finished, status = os.waitpid(child_pid, os.WNOHANG)
            if finished:
                wait_status = status
                break
        if wait_status is None:
            os.kill(child_pid, signal.SIGKILL)
            _, wait_status = os.waitpid(child_pid, 0)
        os.close(master_fd)

        output = transcript.decode("utf-8", errors="replace")
        final_metadata = json.loads(run_metadata.read_text(encoding="utf-8"))
        checks = {
            "confirmation_window_reached": confirmation_seen,
            "workbench_exited_successfully": os.waitstatus_to_exitcode(wait_status) == 0,
            "changed_running_status_rejected": "运行状态 running 已变化或受保护" in output
            or "清理计划已过期" in output,
            "logs_and_progress_preserved": worker_log.read_bytes() == original_log
            and progress.read_bytes() == original_progress,
            "new_run_status_preserved": final_metadata["status"] == "running",
        }
        return {
            "status": "verified" if all(checks.values()) else "failed",
            "requirement_id": "CLI13-clean-plan-status-change",
            "command": "m-model init; PTY cleanup preview; change run status before confirmation; submit fixed plan",
            "exit_code": os.waitstatus_to_exitcode(wait_status),
            "checks": checks,
            "fixture": "temporary project; completed run changed to running while the PTY waited for CLEAN",
            "scope_limits": [
                "Verifies Linux PTY staleness handling for a run status transition only.",
                "Does not verify environment installation or export conflicts.",
            ],
        }, output


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=repository / "target/debug/m-model")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "target/cli13-clean-cancel",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    report, transcript = run(binary, args.python, args.timeout)
    stale_report, stale_transcript = verify_status_change(binary, args.python, args.timeout)
    report["status_change_scenario"] = stale_report
    report["status"] = (
        "verified"
        if report["status"] == "verified" and stale_report["status"] == "verified"
        else "failed"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    (args.output_dir / "status-change-transcript.txt").write_text(
        stale_transcript, encoding="utf-8"
    )
    print(
        f"{report['status']}: CLI-13 cleanup cancel; "
        f"report={args.output_dir / 'report.json'}"
    )
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
