#!/usr/bin/env python3
"""Capture monochrome Drifting graph presets from real TUI tmux panes."""

import argparse
import os
from pathlib import Path
import re
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
    socket = "maven-presets-" + uuid.uuid4().hex[:12]
    base = ["tmux", "-L", socket, "-f", "/dev/null"]

    with tempfile.TemporaryDirectory(prefix="maven-preset-capture-") as directory:
        env = os.environ.copy()
        env.update({
            "TERM": "xterm-256color",
            "LC_ALL": "en_US.UTF-8",
            "MODEL_MAVEN_REDUCE_MOTION": "1",
            "MODEL_MAVEN_GRAPHICS": "cells",
            "MODEL_MAVEN_TUI_PREFERENCES": str(Path(directory) / "preferences.json"),
            "NO_COLOR": "1",
        })
        env.pop("COLUMNS", None)
        env.pop("TMUX", None)

        def tmux(*arguments):
            return subprocess.run(
                base + list(arguments), env=env, text=True, capture_output=True, check=True
            ).stdout

        def frame(session):
            return tmux("capture-pane", "-p", "-t", session)

        def wait_for(session, needle):
            deadline = time.monotonic() + 20
            latest = ""
            while time.monotonic() < deadline:
                latest = frame(session)
                if needle in latest:
                    return latest
                time.sleep(0.05)
            raise RuntimeError(f"Missing {needle!r} in {session} pane:\n{latest}")

        def start(session, *arguments, columns=120, rows=36):
            tmux(
                "new-session", "-d", "-s", session, "-x", str(columns), "-y", str(rows),
                "-c", str(root), str(binary), *arguments,
            )
            tmux("set-option", "-t", session, "status", "off")

        def capture_modes(session, prefix, selection):
            wait_for(session, selection)
            modes = (
                ("1", "spatial", "SPATIAL 3D", "CAM 032°/+30°"),
                ("2", "flow-2d", "2D FLOW", None),
                ("3", "exploded-1", "EXPLODE x1", "CAM 045°/+35°"),
                ("4", "exploded-2", "EXPLODE x2", "CAM 045°/+35°"),
            )
            for key, name, title, camera in modes:
                tmux("send-keys", "-t", session, key)
                captured = wait_for(session, f"{title}  /  MODEL-MAVEN STRUCTURE")
                if selection not in captured:
                    raise RuntimeError(f"Preset {name} lost selection {selection!r}:\n{captured}")
                if prefix in ("drifting", "drift-b") and name.startswith("exploded"):
                    missing = [marker for marker in ("s1", "s2", "s3") if marker not in captured]
                    if missing:
                        raise RuntimeError(
                            f"Preset {name} hides stable stack slice labels {missing}:\n{captured}"
                        )
                if camera and camera not in captured:
                    raise RuntimeError(f"Preset {name} missing camera {camera!r}:\n{captured}")
                marker = {"1": "{1}", "2": "{2}", "3": "{3}", "4": "{4}"}[key]
                if marker not in captured.splitlines()[0]:
                    raise RuntimeError(f"Preset {name} missing active marker {marker!r}")
                if len(captured.splitlines()) not in (35, 36):
                    raise RuntimeError(
                        f"Expected a 120×36 terminal frame for {name}, "
                        f"got {len(captured.splitlines())} rows"
                    )
                (root / "docs/graph-preset-samples" / f"{prefix}-{name}-cells.txt").write_text(
                    captured, encoding="utf-8"
                )

        def open_search_target(session, path):
            tmux("send-keys", "-t", session, "/")
            time.sleep(0.05)
            tmux("send-keys", "-l", "-t", session, path)
            tmux("send-keys", "-t", session, "Enter")
            current = wait_for(session, "SPATIAL 3D  /  MODEL-MAVEN STRUCTURE")
            if path not in current:
                raise RuntimeError(f"Search did not open {path!r}:\n{current}")

        try:
            start("drifting-l", "--drifting-demo")
            capture_modes("drifting-l", "drifting", "DitGen.backbone")

            start(
                "drifting-b",
                "--drifting-source", "./fixtures/drifting",
                "--drifting-config", "./fixtures/drifting/configs/gen/latent_sota_B.yaml",
            )
            capture_modes("drifting-b", "drift-b", "DitGen.backbone")

            start("qkv", "--drifting-demo")
            wait_for("qkv", "DitGen.backbone")
            tmux("send-keys", "-t", "qkv", "/")
            tmux("send-keys", "-l", "-t", "qkv", "block_07.attention.qkv")
            tmux("send-keys", "-t", "qkv", "Enter")
            capture_modes("qkv", "qkv", "DitGen.backbone.blocks.block_07.attention.qkv")

            start("blocks", "--drifting-demo")
            open_search_target("blocks", "DitGen.backbone.blocks.block_07")
            capture_modes("blocks", "drifting-blocks", "DitGen.backbone.blocks.block_07")

            start("residual", "--drifting-demo")
            open_search_target("residual", "DitGen.backbone.blocks.block_07.attention_residual")
            capture_modes(
                "residual", "drifting-residual",
                "DitGen.backbone.blocks.block_07.attention_residual",
            )

            start("narrow", "--drifting-demo", columns=40, rows=24)
            wait_for("narrow", "DitGen.backbone")
            for key, name, mode in (("3", "x1", "exploded slices 1/2"),
                                    ("4", "x2", "exploded slices 2/2")):
                tmux("send-keys", "-t", "narrow", key)
                captured = wait_for("narrow", mode)
                if "DitGen.backbone" not in captured:
                    raise RuntimeError(f"Narrow exploded {name} lost the selected backbone:\n{captured}")
                if len(captured.splitlines()) != 24:
                    raise RuntimeError(f"Expected a 40×24 frame for narrow x{name}: {len(captured.splitlines())}")
                if not re.search(r"\d+–\d+/\d+", captured.splitlines()[1]):
                    raise RuntimeError(f"Narrow exploded {name} hides its page range:\n{captured}")
                (root / "docs/graph-preset-samples" / f"drifting-narrow-exploded-{name}-40x24-cells.txt").write_text(
                    captured, encoding="utf-8"
                )
            print("tmux: captured 20 NO_COLOR 120×36 plus 2 40×24 Drifting preset frames and verified camera angles")
        finally:
            subprocess.run(
                base + ["kill-server"], env=env, text=True, capture_output=True, check=False
            )


if __name__ == "__main__":
    main()
