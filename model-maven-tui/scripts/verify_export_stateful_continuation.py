#!/usr/bin/env python3
"""Saved exported continuations restore stateful data and RNG in new processes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-export-stateful-") as directory:
    parent = Path(directory)
    project = parent / "project"; project.mkdir()
    exported = parent / "export"
    (project / "stateful.py").write_text('''import random
import torch
cursor = 0
def batch():
    global cursor
    cursor += 1
    value = torch.rand(2, 2) + random.random() + cursor
    return {"args": [], "kwargs": {"x": value}, "loss_inputs": {"target": torch.ones(2, 1)}}
def save(): return {"cursor": cursor}
def load(state):
    global cursor
    cursor = state["cursor"]
batch.state_dict = save
batch.load_state_dict = load
''')
    env = os.environ.copy(); env.pop("PYTHONPATH", None); env.pop("PYTHONHOME", None)
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        assert result.returncode == 0, report
        return report
    args = ("scaffold", "--input", "x=2", "--output-dim", "1")
    preview = run(*args); run(*args, "--apply", "--if-plan", preview["data"]["plan_revision"])
    args = ("config", "set-batch", "--batch", "stateful:batch", "--batch-binding", "envelope")
    preview = run(*args); run(*args, "--apply", "--if-current", preview["data"]["from_revision"])
    original = run("train", "--step", "model_starter:step", "--steps", "1", "--checkpoint-every", "1")
    run_id = original["data"]["run_id"]
    run("export", str(exported), "--run", run_id)
    reference = run("train", "--resume", run_id, "--additional-steps", "2")
    expected = reference["data"]["model_state_sha256"]
    hidden = project.with_name("hidden")
    project.rename(hidden)
    try:
        first = parent / "first"
        second = parent / "second"
        script_one = "import restore_model as r,sys; o=r.resume_managed_steps(1,checkpoint_directory=sys.argv[1]); print(o['continuation_checkpoint']['sha256'])"
        result = subprocess.run([sys.executable, "-c", script_one, str(first)], cwd=exported, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (result.stdout, result.stderr)
        script_two = "import restore_model as r,sys; o=r.resume_managed_steps(1,from_checkpoint=sys.argv[1],checkpoint_directory=sys.argv[2]); assert r.worker.model_state_digest(r.model)==sys.argv[3]; print('MATCH')"
        result = subprocess.run([sys.executable, "-c", script_two, str(first), str(second), expected], cwd=exported, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0 and "MATCH" in result.stdout, (result.stdout, result.stderr)
        check = "import json,sys,torch; from pathlib import Path; d=Path(sys.argv[1]); x=json.loads((d/'latest.json').read_text()); p=torch.load(d/x['file'],map_location='cpu',weights_only=True); assert p['completed_steps']==3; assert p['batch_state']['kind']=='hooks' and p['batch_state']['state']['cursor']==3; assert p['engine_state']['optimizer_state']['state']; print('STATE_OK')"
        result = subprocess.run([sys.executable, "-c", check, str(second)], cwd=exported, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0 and "STATE_OK" in result.stdout, (result.stdout, result.stderr)
        wrong = parent / "wrong-bundle"; shutil.copytree(first, wrong)
        tamper = "import hashlib,json,sys,torch; from pathlib import Path; d=Path(sys.argv[1]); m=json.loads((d/'latest.json').read_text()); p=d/m['file']; s=torch.load(p,map_location='cpu',weights_only=True); s['export_continuation']['base_manifest_sha256']='0'*64; torch.save(s,p); m['sha256']=hashlib.sha256(p.read_bytes()).hexdigest(); (d/'latest.json').write_text(json.dumps(m))"
        subprocess.run([sys.executable, "-c", tamper, str(wrong)], cwd=exported, env=env, check=True, capture_output=True, timeout=120)
        rejected_dir = parent / "must-not-exist"
        rejected = subprocess.run([sys.executable, "-c", script_two, str(wrong), str(rejected_dir), expected], cwd=exported, env=env, capture_output=True, text=True, timeout=120)
        assert rejected.returncode != 0 and "不属于此导出包" in rejected.stderr and not rejected_dir.exists(), (rejected.stdout, rejected.stderr)
        print("Stateful batch cursor, RNG and optimizer persist across independent continuation processes")
    finally:
        hidden.rename(project)
