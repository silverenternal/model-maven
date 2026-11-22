#!/usr/bin/env python3
"""Run release graph benchmarks and enforce this host's provisional TUI budget."""

import os
import platform
import re
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DURATION = r"([0-9.]+)(ms|µs)"


def run(command):
    environment = os.environ.copy()
    environment["RUSTC_WRAPPER"] = ""
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    sys.stdout.write(result.stdout)
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout


def to_ms(value, unit):
    value = float(value)
    return value if unit == "ms" else value / 1000


def metric_ms(match, value_group, unit_group):
    return to_ms(match.group(value_group), match.group(unit_group))


def require(label, actual, limit, unit="ms"):
    print(f"budget {label}: {actual:.3f} {unit} (limit {limit} {unit})")
    if actual > limit:
        raise SystemExit(f"performance budget exceeded: {label}")


def main():
    if platform.system() != "Linux":
        raise SystemExit("this provisional performance budget is Linux-only")

    render_output = run(
        [
            "cargo",
            "test",
            "--release",
            "--locked",
            "--offline",
            "thousand_operator_graph_keeps_layout_and_rendering_local",
            "--",
            "--nocapture",
        ]
    )
    pty_output = run(
        [
            "cargo",
            "test",
            "--release",
            "--locked",
            "--offline",
            "--test",
            "welcome",
            "large_offline_workbench_measures_input_and_retains_each_edit",
            "--",
            "--nocapture",
        ]
    )

    frames = {}
    for line in render_output.splitlines():
        match = re.search(
            rf"(\d+)×(\d+) frame, pixels=(true|false): "
            rf"total p50 {DURATION}, p95 {DURATION}, .* max (\d+) bytes",
            line,
        )
        if match:
            size = (int(match.group(1)), int(match.group(2)))
            mode = "pixels" if match.group(3) == "true" else "cells"
            frames[size, mode] = match

    for mode, budget in [("cells", 1.0), ("pixels", 5.0)]:
        frame = frames.get(((200, 60), mode))
        if frame is None:
            raise SystemExit(f"missing 200×60 {mode} frame metric")
        require(
            f"200×60 {mode} frame p95",
            metric_ms(frame, 6, 7),
            budget,
        )
        require(f"200×60 {mode} frame bytes", int(frame.group(8)), 20_000, "B")

    for mode, first_limit, response_limit, burst_limit in [
        ("cells", 15.0, 2.0, 10.0),
        ("pixels", 25.0, 25.0, 30.0),
    ]:
        first = re.search(
            rf"100×30 {mode}, 1024 generated layers: first frame {DURATION};",
            pty_output,
        )
        response = re.search(
            rf"100×30 {mode}, 1024 generated layers: first frame {DURATION}; "
            rf"100 events: PTY response p50 {DURATION}, p95 {DURATION},",
            pty_output,
        )
        activity = re.search(
            rf"{mode}: idle ([0-9.]+)ms, (\d+) bytes; details only {DURATION}, \d+ bytes; "
            rf"burst to v2 {DURATION}, v3 {DURATION}; rejection \d+ bytes",
            pty_output,
        )
        if first is None or response is None or activity is None:
            raise SystemExit(f"missing 100×30 {mode} PTY metrics")

        require(f"100×30 {mode} first frame", metric_ms(first, 1, 2), first_limit)
        require(
            f"100×30 {mode} input response p95",
            metric_ms(response, 5, 6),
            response_limit,
        )

        idle_ms = float(activity.group(1))
        idle_bytes = int(activity.group(2))
        if idle_ms < 80 or idle_bytes != 0:
            raise SystemExit(
                f"idle budget failed for {mode}: {idle_ms:.1f}ms, {idle_bytes} bytes"
            )
        print(f"budget 100×30 {mode} idle: {idle_ms:.1f}ms, {idle_bytes} bytes")

        require(
            f"100×30 {mode} commit to v3",
            metric_ms(activity, 7, 8),
            burst_limit,
        )


if __name__ == "__main__":
    main()
