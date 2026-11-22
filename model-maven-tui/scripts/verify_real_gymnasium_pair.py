#!/usr/bin/env python3
"""Exercise a user-owned Gymnasium loop in an explicitly selected environment.

Run with a pre-existing venv containing both Gymnasium and PyTorch. Nothing is
installed by this script and the fixture environment is isolated in a temp dir.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import gymnasium
import torch

root = Path(__file__).resolve().parent.parent
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
source_env = Path(sys.prefix)
assert source_env != Path(sys.base_prefix), "run with a pre-existing venv"
with tempfile.TemporaryDirectory(prefix="maven-real-gym-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    (project / "rl_fixture.py").write_text('''import gymnasium as gym
import torch
from pathlib import Path
from torch import nn
def model(): return nn.Linear(4, 2)
def batch(): return torch.ones(2, 4)
def loop(model, context):
    environment = gym.make("CartPole-v1")
    try:
        observation, info = environment.reset(seed=42)
        environment.action_space.seed(42)
        Path("gym-loop-ran").write_text(gym.__version__)
        while context.remaining_steps:
            observation, reward, terminated, truncated, info = environment.step(environment.action_space.sample())
            if not context.report_step(torch.tensor(-reward), {"reward": float(reward)}): break
            if terminated or truncated: observation, info = environment.reset()
    finally:
        environment.close()
''')
    def command(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        assert result.returncode == 0, report
        return report
    command("init", "--model", "rl_fixture:model", "--batch", "rl_fixture:batch")
    command("env", "configure", "--system-site-packages", "--venv", ".venv")
    shutil.copytree(source_env, project / ".venv", symlinks=True)
    config = (project / ".model-maven/environment.json").read_bytes()
    inventory = command("env", "libraries", "--library", "gymnasium")
    library = next(item for item in inventory["data"]["libraries"] if item["distribution"] == "gymnasium")
    assert library["installed"] is True and library["auto_bound"] is False and library["auto_installed"] is False, inventory
    assert not (project / "gym-loop-ran").exists()
    probe = command("env", "probe", "--library", "gymnasium", "--check-runtime")
    assert probe["data"]["auto_bound"] is False and probe["data"]["auto_installed"] is False, probe
    assert probe["data"]["probes"][0]["status"] == "imported", probe
    assert probe["data"]["probes"][0]["runtime_check"]["status"] == "passed", probe
    inputs = "\nr\ntrain\nexternal-loop\nrl_fixture:loop\n{}\n2\n\nn\ny\nq\n"
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                            env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    assert "Mode: external-loop" in result.stdout and "Run started or completed" in result.stdout, result.stdout
    listing = command("runs", "list")["data"]["runs"]
    run = next(item for item in listing if item["operation"] == "train")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        detail = command("runs", "show", run["id"])["data"]
        if detail["record"]["status"] not in ("running", "stopping"): break
        time.sleep(.2)
    assert detail["record"]["status"] == "completed", detail
    assert (project / "gym-loop-ran").read_text() == gymnasium.__version__
    assert detail["report"]["data"]["completed_steps"] == 2, detail
    assert (project / ".model-maven/environment.json").read_bytes() == config
    print(json.dumps({"gymnasium":gymnasium.__version__, "torch":str(torch.__version__),
                      "mode":"user-owned external-loop", "steps":2,
                      "automatic_binding":False, "scope":"CartPole CPU fixture"}))
