#!/usr/bin/env python3
"""Replay the crossed-port keyboard path in an isolated headless tmux server."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=root / "target/release/m-model-tui")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    socket = "maven-audit-" + uuid.uuid4().hex[:12]
    base = ["tmux", "-L", socket, "-f", "/dev/null"]

    with tempfile.TemporaryDirectory(prefix="maven-tmux-") as directory:
        env = os.environ.copy()
        env.update({
            "MODEL_MAVEN_REDUCE_MOTION": "1",
            "MODEL_MAVEN_GRAPHICS": "cells",
            "MODEL_MAVEN_TUI_PREFERENCES": str(Path(directory) / "preferences.json"),
            "NO_COLOR": "1",
            "LC_ALL": "en_US.UTF-8",
        })
        env.pop("COLUMNS", None)
        env.pop("TMUX", None)

        def tmux(*arguments, check=True):
            return subprocess.run(base + list(arguments), env=env, text=True,
                                  capture_output=True, check=check)

        def frame(session="audit"):
            return tmux("capture-pane", "-p", "-t", session).stdout

        def wait_for(needle, session="audit"):
            deadline = time.monotonic() + 15
            latest = ""
            while time.monotonic() < deadline:
                latest = frame(session)
                if needle in latest:
                    return latest
                time.sleep(0.05)
            raise RuntimeError(f"Missing {needle!r} in tmux pane:\n{latest}")

        def wait_for_state(required, forbidden=(), session="audit"):
            deadline = time.monotonic() + 15
            latest = ""
            while time.monotonic() < deadline:
                latest = frame(session)
                if all(item in latest for item in required) and not any(
                    item in latest for item in forbidden
                ):
                    return latest
                time.sleep(0.05)
            raise RuntimeError(
                f"Missing graph state {required!r} or found {forbidden!r}:\n{latest}"
            )

        try:
            tmux("new-session", "-d", "-s", "audit", "-x", "100", "-y", "30",
                 "-c", str(root), str(binary), "--diagnosis-demo", "--diagnosis-case", "crossed-ports")
            tmux("set-option", "-g", "remain-on-exit", "on")
            wait_for("Open workbench")
            tmux("send-keys", "-t", "audit", "Enter")
            initial = wait_for("model.head #2")
            if "Observed 2×48" not in initial:
                raise RuntimeError("Missing observed 48/32 diagnosis")
            tmux("send-keys", "-t", "audit", "Tab")
            wait_for("Tab Details")
            preset_frames = {}
            preset_names = {
                "1": "spatial",
                "2": "flow-2d",
                "3": "exploded-x1",
                "4": "exploded-x2",
            }
            for key, label in (("4", ("exploded slices 2/2",)),
                               ("3", ("exploded slices 1/2",)),
                               ("1", ("CAM ",)) ,
                               ("2", ("2D  ·  OFFLINE",))):
                tmux("send-keys", "-t", "audit", key)
                if key == "1":
                    current = wait_for_state(label, ("exploded slices",))
                else:
                    current = wait_for_state(label)
                selected = {
                    "1": "{1}[2][3][4]",
                    "2": "[1]{2}[3][4]",
                    "3": "[1][2]{3}[4]",
                    "4": "[1][2][3]{4}",
                }[key]
                if selected not in current:
                    raise RuntimeError(f"Preset {key} missing monochrome active marker {selected!r}:\n{current}")
                header = current.splitlines()[0]
                if (
                    "? Controls" not in header
                    or "M Port matrix" not in header
                    or "T Mouse on" not in header
                ):
                    raise RuntimeError(f"Preset toolbar displaced graph controls or mouse state: {header}")
                orientation = {
                    "1": "CAM 032°/+30°",
                    "3": "CAM 045°/+35°",
                    "4": "CAM 045°/+35°",
                }.get(key)
                if orientation and orientation not in current:
                    raise RuntimeError(f"Preset {key} missing expected camera {orientation!r}:\n{current}")
                if len(current.splitlines()) != 30:
                    raise RuntimeError(f"Expected a 100×30 frame for preset {key}, got {len(current.splitlines())} rows")
                preset_frames[key] = current
            samples = root / "docs/graph-preset-samples"
            for key, captured in preset_frames.items():
                name = preset_names[key]
                (samples / f"toolbar-{name}-100x30-cells.txt").write_text(captured, encoding="utf-8")
            routes = set()
            expected = {"model.encoder[1] → net.merge[0]", "model.encoder[0] → net.merge[1]"}
            route_frames = {}
            for _ in range(4):
                tmux("send-keys", "-t", "audit", "w")
                time.sleep(0.08)
                current = frame()
                routes.update(route for route in expected if route in current)
                route_frames.update({route: current for route in expected if route in current})
            if routes != expected:
                raise RuntimeError(f"Missing crossed routes: {expected - routes}")
            braille = lambda captured: "".join(ch for ch in captured if "\u2801" <= ch <= "\u28ff")
            first_route, second_route = sorted(expected)
            if braille(route_frames[first_route]) == braille(route_frames[second_route]):
                raise RuntimeError("Pinning crossed routes did not change the monochrome graph raster")
            for route, suffix in zip(sorted(expected), ("a", "b")):
                (samples / f"crossed-ports-selected-{suffix}-100x30-cells.txt").write_text(
                    route_frames[route], encoding="utf-8"
                )
            tmux("send-keys", "-t", "audit", "/")
            wait_for("[1/z]")
            wait_for("[M]")
            tmux("send-keys", "-t", "audit", "spacing")
            wait_for("[3/4] Spread to exploded x1")
            tmux("send-keys", "-t", "audit", "Enter")
            wait_for("exploded slices 1/2")
            tmux("send-keys", "-t", "audit", "1")
            wait_for_state(("CAM ",), ("exploded slices",))
            tmux("send-keys", "-t", "audit", "2")
            wait_for_state(("2D  ·  OFFLINE",))
            tmux("send-keys", "-t", "audit", "/")
            tmux("send-keys", "-t", "audit", "C-u")
            wait_for("[1/z]")
            tmux("send-keys", "-t", "audit", "Escape")
            wait_for("Space · Toggle motion")
            tmux("send-keys", "-t", "audit", "q")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if tmux("display-message", "-p", "-t", "audit", "#{pane_dead}").stdout.strip() == "1":
                    status = tmux("display-message", "-p", "-t", "audit", "#{pane_dead_status}").stdout.strip()
                    if status != "0":
                        raise RuntimeError(f"TUI exited with status {status}")
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("TUI did not exit after q")

            tmux("new-session", "-d", "-s", "narrow", "-x", "40", "-y", "24",
                 "-c", str(root), str(binary), "--diagnosis-demo", "--diagnosis-case", "last-observed")
            tmux("set-option", "-t", "narrow", "status", "off")
            wait_for("Open workbench", "narrow")
            tmux("send-keys", "-t", "narrow", "Enter")
            wait_for("model.head #2", "narrow")
            tmux("send-keys", "-t", "narrow", "Tab")
            wait_for("Tab Details", "narrow")
            tmux("send-keys", "-t", "narrow", "4")
            narrow_x2 = wait_for_state(("exploded slices 2/2", "4 x2"), session="narrow")
            if len(narrow_x2.splitlines()) != 24:
                raise RuntimeError(f"Expected a 40×24 pane, got {len(narrow_x2.splitlines())} rows")
            tmux("send-keys", "-t", "narrow", "3")
            wait_for_state(("exploded slices 1/2", "3 x1"), session="narrow")
            print("tmux: 100×30 visible 1–4 preset markers/search and crossed routes; 40×24 x1/x2 hints verified")
        finally:
            tmux("kill-server", check=False)


if __name__ == "__main__":
    main()
