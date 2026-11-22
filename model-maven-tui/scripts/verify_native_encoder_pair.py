#!/usr/bin/env python3
"""Build and train a variable-length model whose fusion consumes native encoder output."""
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-native-encoder-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"), COLUMNS="160")
    (project / "components.py").write_text('''import torch
from torch import nn
class Fusion(nn.Module):
    def forward(self, values):
        sequence, tabular = values
        assert sequence.dtype == torch.int64 and sequence.ndim == 3 and sequence.shape[2] == 4
        return torch.cat((sequence.float().mean(dim=1), tabular), dim=1)
def fusion(): return Fusion()
def head(): return nn.Linear(6, 3)
''')
    (project / "factor.txt").write_text("1")
    (project / "data.py").write_text('''from pathlib import Path
import model_starter
def batch():
    value = model_starter.batch()
    value["kwargs"]["text"] *= int(Path("factor.txt").read_text())
    return value
''')
    inputs = ("\nm\ntext:int64=?5,4;tabular=2;@native=text_encoder;"
              "@batch=data:batch;@seam.fusion=components:fusion;@seam.head=components:head\n3\ny\nq\n")
    proc = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                          env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    assert "Native encoder outputs (no automatic reduction or flattening)" in proc.stdout, proc.stdout
    assert "Custom model project created." in proc.stdout, proc.stdout
    def run(*args, success=True):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        assert (proc.returncode == 0) == success, report
        return report
    assert 'self.native_encoder_outputs = ["text_encoder"]' in (project / "model_starter.py").read_text()
    report = run("scaffold", "--input", "x=4", "--native-encoder", "missing_encoder", success=False)
    assert report["diagnostics"][0]["code"] == "SCAFFOLD_NATIVE_ENCODER_INVALID", report
    run("model", "probe", "--batch", "model_starter:batch", "--batch-param", "text_dim1=7")
    trial = run("try", "--step", "model_starter:step", "--steps", "1")
    assert trial["data"]["completed_steps"] == 1, trial
    report = run("assembly", "generate")
    assert report["data"]["generated_validation"]["forward"] == "passed", report
    restored = subprocess.run(["python3", "-c", "from generated.model_assembly import build_model; import model_starter; m=build_model(); assert m(**model_starter.batch(text_dim1=7)['kwargs']).shape == (2,3)"],
                              cwd=project, env=env, capture_output=True, text=True, timeout=60)
    assert restored.returncode == 0, (restored.stdout, restored.stderr)
    config = project / ".model-maven/project.json"
    original = config.read_bytes()
    def session(text, executable=cli):
        return subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(executable)], cwd=project,
                              env=env, input=text, capture_output=True, text=True, timeout=180, check=True).stdout
    failed = session("\na\nencoder-output\ntext_encoder\nflatten\nq\n")
    assert "FORWARD_FAILED" in failed and config.read_bytes() == original, failed
    (project / "components.py").write_text('''import torch
from torch import nn
class Fusion(nn.Module):
    def forward(self, values):
        sequence, tabular = values
        if sequence.ndim == 3:
            assert sequence.dtype == torch.int64
            sequence = sequence.float().mean(dim=1)
        else:
            assert sequence.ndim == 2 and sequence.dtype == torch.float32
        return torch.cat((sequence, tabular), dim=1)
def fusion(): return Fusion()
def head(): return nn.Linear(6, 3)
''')
    cancelled = session("\na\nencoder-output\ntext_encoder\nflatten\nn\nn\nq\n")
    assert "Encoder output preview" in cancelled and config.read_bytes() == original, cancelled
    wrapper = project / "race-cli"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "encoder-output" in args and "--if-config-revision" in args:
    path = Path(".model-maven/project.json")
    value = json.loads(path.read_text()); value["concurrent_marker"] = "preserved"
    path.write_text(json.dumps(value))
os.execv({cli!r}, [{cli!r}, *args])
''')
    wrapper.chmod(0o755)
    raced = session("\na\nencoder-output\ntext_encoder\nflatten\nn\ny\nq\n", wrapper)
    assert "PROJECT_CONFIG_CONFLICT" in raced, raced
    assert json.loads(config.read_text())["concurrent_marker"] == "preserved"
    source_before = (project / "components.py").read_bytes()
    source_race = project / "source-race-cli"
    source_race.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
if "encoder-output" in args and "--if-source-revision" in args:
    with Path("components.py").open("a") as stream: stream.write("\\n# concurrent change\\n")
os.execv({cli!r}, [{cli!r}, *args])
''')
    source_race.chmod(0o755)
    original_config = config.read_bytes()
    raced = session("\na\nencoder-output\ntext_encoder\nflatten\nn\ny\nq\n", source_race)
    assert "SOURCE_MANIFEST_CONFLICT" in raced and config.read_bytes() == original_config, raced
    (project / "components.py").write_bytes(source_before)
    input_race = project / "input-race-cli"
    input_race.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
if "encoder-output" in args and "--if-input-sha256" in args:
    Path("factor.txt").write_text("2")
os.execv({cli!r}, [{cli!r}, *args])
''')
    input_race.chmod(0o755)
    raced = session("\na\nencoder-output\ntext_encoder\nflatten\ny\ny\nq\n", input_race)
    assert "INPUT_CONTENT_MISMATCH" in raced and config.read_bytes() == original_config, raced
    (project / "factor.txt").write_text("1")
    saved = session("\na\nencoder-output\ntext_encoder\nflatten\ny\ny\nq\n")
    assert "Encoder output preview" in saved, saved
    assert json.loads(config.read_text())["model_params"]["encoder_output_modes"]["text_encoder"] == "flatten", saved
    run("model", "probe", "--batch", "model_starter:batch", "--batch-param", "text_dim1=7")
    session("\na\nencoder-output\ntext_encoder\nnative\ny\ny\nq\n")
    assert json.loads(config.read_text())["model_params"]["encoder_output_modes"]["text_encoder"] == "native"
    run("model", "probe", "--batch", "model_starter:batch", "--batch-param", "text_dim1=7")
    (project / "data.py").write_text('''from pathlib import Path
import model_starter
def batch():
    path = Path("factor.txt")
    factor = int(path.read_text())
    path.write_text(str(factor + 1))
    value = model_starter.batch()
    value["kwargs"]["text"] *= factor
    return value
''')
    (project / "factor.txt").write_text("1")
    saved = session("\na\nencoder-output+snapshot\ntext_encoder\nflatten\ny\nq\n")
    match = re.search(r"Temporary file: ([^\s]+)", saved)
    assert match and not Path(match.group(1)).exists() and "Temporary input snapshot removed" in saved, saved
    assert json.loads(config.read_text())["model_params"]["encoder_output_modes"]["text_encoder"] == "flatten", saved
    assert int((project / "factor.txt").read_text()) == 2, saved
    print("TUI preserves variable-length int64 encoder output for custom fusion; probe, training and fresh-process reconstruction succeed")
    print("TUI rejects incompatible flatten, preserves cancelled preview, and switches flatten/native after compatible fusion is supplied")
    print("Configuration, observed source, and optionally input changes after preview prevent the save")
    print("Dynamic encoder mode preview replays its captured input and removes the temporary snapshot")
