#!/usr/bin/env python3
"""Create, diagnose, conditionally edit and rebuild actual model input routes."""
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-route-edit-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), REAL_MAVEN_CLI=cli, COLUMNS="160")
    def run(*args, ok=True):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        assert (result.returncode == 0) == ok, report
        return report
    def session(inputs, executable=cli):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(executable)], cwd=project,
                              env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True).stdout
    (project / "factor.txt").write_text("1")
    (project / "data.py").write_text('''from pathlib import Path
import model_starter
def batch():
    value = model_starter.batch()
    value["kwargs"]["a"] *= float(Path("factor.txt").read_text())
    return value
batch.MODEL_MAVEN_STATELESS = True
''')
    args = ("scaffold", "--input", "a=4", "--input", "b=2", "--output-dim", "3", "--batch", "data:batch")
    plan = run(*args)
    run(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    (project / "components.py").write_text("from torch import nn\ndef encoder(): return nn.Linear(4, 4)\n")
    run("assembly", "add", "a_encoder", "--factory", "components:encoder")
    source = (project / "model_starter.py").read_bytes()
    config_path = project / ".model-maven/project.json"
    original = config_path.read_bytes()
    failed = session("\na\nroute\na_encoder\nb\nq\n")
    assert "FORWARD_FAILED" in failed and config_path.read_bytes() == original, failed
    cancelled = session("\na\nroute\nb_encoder\na\nn\nn\nq\n")
    assert "Input route preview" in cancelled and config_path.read_bytes() == original, cancelled
    wrapper = project / "race-cli"
    wrapper.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if 'route' in args and '--if-config-revision' in args:
    path = Path('.model-maven/project.json')
    config = json.loads(path.read_text()); config['concurrent_marker'] = 'preserved'
    path.write_text(json.dumps(config))
real = os.environ['REAL_MAVEN_CLI']
os.execv(real, [real] + args)
''')
    wrapper.chmod(0o755)
    raced = session("\na\nroute\nb_encoder\na\nn\ny\nq\n", wrapper)
    assert "PROJECT_CONFIG_CONFLICT" in raced, raced
    assert json.loads(config_path.read_text())["concurrent_marker"] == "preserved"
    original_config = config_path.read_bytes()
    source_before = (project / "components.py").read_bytes()
    source_race = project / "source-race-cli"
    source_race.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
if "route" in args and "--if-source-revision" in args:
    with Path("components.py").open("a") as stream: stream.write("\\n# concurrent source change\\n")
real = os.environ["REAL_MAVEN_CLI"]
os.execv(real, [real] + args)
''')
    source_race.chmod(0o755)
    raced = session("\na\nroute\nb_encoder\na\nn\ny\nq\n", source_race)
    assert "SOURCE_MANIFEST_CONFLICT" in raced and config_path.read_bytes() == original_config, raced
    (project / "components.py").write_bytes(source_before)
    input_race = project / "input-race-cli"
    input_race.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
if "route" in args and "--if-input-sha256" in args:
    Path("factor.txt").write_text("2")
real = os.environ["REAL_MAVEN_CLI"]
os.execv(real, [real] + args)
''')
    input_race.chmod(0o755)
    raced = session("\na\nroute\nb_encoder\na\ny\ny\nq\n", input_race)
    assert "INPUT_CONTENT_MISMATCH" in raced and config_path.read_bytes() == original_config, raced
    (project / "factor.txt").write_text("1")
    before_revision = run("assembly", "show", "--config-only")["data"]["config_revision"]
    saved = session("\na\nroute\nb_encoder\na\ny\ny\n2\n\n\nq\n")
    assert "Input route preview" in saved and "Observed calls for this batch" in saved, saved
    config = json.loads(config_path.read_text())
    assert config["model_params"]["input_routes"] == {"b_encoder": "a"}, config
    assert config["concurrent_marker"] == "preserved"
    assert (project / "model_starter.py").read_bytes() == source
    training_plan = session("\nr\ntry-wait\n\n\n\n\n\nn\nn\nq\n")
    assert "Project model parameters" in training_plan and '"b_encoder":"a"' in training_plan, training_plan
    probe = run("model", "probe", "--model", "model_starter:model", "--batch", "model_starter:batch")
    assert probe["status"] == "success", probe
    run("try", "--step", "model_starter:step", "--steps", "1")
    trained = run("train", "--step", "model_starter:step", "--steps", "1", "--checkpoint-every", "1")
    trained_id = trained["data"]["run_id"]
    assert run("runs", "show", trained_id)["data"]["inputs"]["model_params"]["input_routes"] == {"b_encoder": "a"}
    generated = run("assembly", "generate")
    assert generated["data"]["validated"]["forward"] == "passed", generated
    script = "from generated.model_assembly import build_model\nimport model_starter\nm=build_model(); assert m.input_routes['b_encoder']=='a'; assert m.head.in_features==8\nassert m(**model_starter.batch()['kwargs']).shape==(2,3)\n"
    process = subprocess.run(["python3", "-c", script], cwd=project, env=env, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, (process.stdout, process.stderr)
    restore = run("config", "restore", before_revision, "--field", "model_params")
    run("config", "restore", before_revision, "--field", "model_params", "--apply", "--if-current", restore["data"]["from_revision"])
    assert not json.loads(config_path.read_text()).get("model_params")
    assert run("model", "probe")["status"] == "success"
    resumed = run("train", "--resume", trained_id, "--additional-steps", "1")
    resumed_id = resumed["data"]["run_id"]
    assert run("runs", "show", resumed_id)["data"]["inputs"]["model_params"]["input_routes"] == {"b_encoder": "a"}
    (project / "data.py").write_text('''from pathlib import Path
import model_starter
def batch():
    path = Path("factor.txt")
    factor = float(path.read_text())
    path.write_text(str(factor + 1))
    value = model_starter.batch()
    value["kwargs"]["a"] *= factor
    return value
batch.MODEL_MAVEN_STATELESS = True
''')
    (project / "factor.txt").write_text("1")
    before_snapshot = config_path.read_bytes()
    invalid_replay = run("assembly", "route", "b_encoder", "a", "--replay-input-from", "missing.snapshot", ok=False)
    assert invalid_replay["diagnostics"][0]["code"] == "INPUT_SNAPSHOT_UNSUPPORTED", invalid_replay
    failed = session("\na\nroute+snapshot\na_encoder\nb\nq\n")
    match = re.search(r"Temporary file: ([^\s]+)", failed)
    assert match and not Path(match.group(1)).exists() and "Temporary input snapshot removed" in failed and "FORWARD_FAILED" in failed, failed
    assert config_path.read_bytes() == before_snapshot and float((project / "factor.txt").read_text()) == 2
    (project / "factor.txt").write_text("1")
    cancelled = session("\na\nroute+snapshot\nb_encoder\na\nn\nq\n")
    match = re.search(r"Temporary file: ([^\s]+)", cancelled)
    assert match and not Path(match.group(1)).exists() and "Temporary input snapshot removed" in cancelled, cancelled
    assert config_path.read_bytes() == before_snapshot and float((project / "factor.txt").read_text()) == 2, (cancelled, (project / "factor.txt").read_text())
    (project / "factor.txt").write_text("1")
    saved = session("\na\nroute+snapshot\nb_encoder\na\ny\nq\n")
    match = re.search(r"Temporary file: ([^\s]+)", saved)
    assert match and not Path(match.group(1)).exists() and "Temporary input snapshot removed" in saved, saved
    assert json.loads(config_path.read_text())["model_params"]["input_routes"]["b_encoder"] == "a"
    assert float((project / "factor.txt").read_text()) == 2, saved
    print("Dynamic route preview captures once, replays on commit and removes snapshots on save and cancel")
    print("Bad route and changed config, observed source or selected input are rejected; guarded route runs, rebuilds, restores and resumes")
