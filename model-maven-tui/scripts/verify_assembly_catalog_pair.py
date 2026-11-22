#!/usr/bin/env python3
"""Choose and parameterize a registered layer during real model assembly."""
import json
import os
import fcntl
import pty
import select
import struct
import termios
import time
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-assembly-catalog-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"))
    (project / "local_components.py").write_text('''import torch
from torch import nn
def make(negative_slope=0.1, tag="small"):
    return nn.LeakyReLU(negative_slope=negative_slope)
def tabular(): return nn.Sequential(nn.Linear(5, 5), nn.ReLU())
class Fusion(nn.Module):
    def forward(self, values): return torch.cat(values, dim=1)
def fusion(): return Fusion()
def head(): return nn.Linear(53, 3)
''')
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120, check=True)
        return json.loads(result.stdout)
    args = ("scaffold", "--input", "image=3,4,4", "--input", "tabular=5", "--output-dim", "3")
    preview = run(*args)
    run(*args, "--apply", "--if-plan", preview["data"]["plan_revision"])
    schema = {"type": "object", "properties": {"negative_slope": {"type": "number", "minimum": 0, "maximum": 1}, "tag": {"type": "string", "enum": ["small", "big"]}}, "required": ["negative_slope", "tag"], "additionalProperties": False}
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 110, 0, 0))
    registration = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                                    env=env, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    pending = bytearray()
    def until(token, timeout=60):
        needle, deadline = token.encode(), time.monotonic() + timeout
        while time.monotonic() < deadline:
            position = pending.find(needle)
            if position >= 0:
                end = position + len(needle)
                result = bytes(pending[:end]).decode(errors="replace")
                del pending[:end]
                return result
            if registration.poll() is not None: raise AssertionError(f"TUI exited: {pending[-4000:]!r}")
            if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-4000:]!r}")
    def send(value): os.write(master, value.encode())
    menu = "list, search, select NUMBER"
    try:
        until("Open workbench"); send("\nz\n"); until(menu); send("add\n")
        until("Component ID"); send("demo/activation\n")
        until("Kind:"); send("\n")
        until("Python factory"); send("local_components:make\n")
        until("Component version"); send("1.2.3\n")
        until("Description (optional)"); send("Trainable activation\n")
        until("Configure schema"); send("y\n")
        until("Parameter schema JSON"); send(json.dumps(schema) + "\n")
        until("Capabilities JSON"); send("\n")
        until("Dependencies JSON"); send("\n")
        until("Conformance fixture"); send("\n")
        until("Type apply to continue"); send("apply\n"); until(menu)
        for _ in range(100):
            send("refresh\n")
            page = until(menu)
            if "Registered and selected: demo/activation@1.2.3" in page:
                break
            time.sleep(0.3)
        else: raise AssertionError(f"registration did not succeed: {page}")
        send("b\n"); until("Model-Maven Workbench"); send("q\n")
        registration.wait(timeout=10)
        assert registration.returncode == 0
    finally:
        if registration.poll() is None:
            registration.terminate(); registration.wait(timeout=10)
        os.close(master)
    # Expand only the list response; every selected real component operation
    # continues to execute against the backend and its actual registry.
    wrapper = project / "large-catalog.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import copy, json, os, subprocess, sys
if "component" in sys.argv and "list" in sys.argv:
    result = subprocess.run([{cli!r}] + sys.argv[1:], capture_output=True, text=True)
    report = json.loads(result.stdout)
    actual = report["data"]["components"]
    extra = []
    for i in range(70):
        item = copy.deepcopy(actual[0])
        item["manifest"]["component_id"] = f"synthetic/component-{{i:02d}}"
        extra.append(item)
    report["data"]["components"] = extra + actual
    print(json.dumps(report))
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    search = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                            env=env, input="\na\n\nimage_encoder\n:catalog\n/activation\n1\nb\nq\n",
                            capture_output=True, text=True, timeout=90, check=True)
    assert "1 matches" in search.stdout and "demo/activation@1.2.3" in search.stdout, search.stdout
    inputs = ("\na\n\nimage_encoder\n:catalog\nn\nn\n71\n\n2\n0.25\nbig\ny\n"
              "a\n\ntabular_encoder\nlocal_components:tabular\ny\n"
              "a\n\nfusion\nlocal_components:fusion\ny\n"
              "a\n\nhead\nlocal_components:head\ny\n2\n\n\na\ngenerate\n\ny\nq\n")
    result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                            env=env, input=inputs, capture_output=True, text=True, timeout=180, check=True)
    output = result.stdout
    assert "Registered Layer components" in output and "demo/activation@1.2.3" in output, output
    assert "3/3" in output and "71. demo/activation@1.2.3" in output, output
    assert "above the maximum" in output and "Assembly preview" in output, output
    assert "Assembly saved" in output and "LeakyReLU" in output, output
    assembly = run("assembly", "show", "--config-only")["data"]
    slot = next(item for item in assembly["slots"] if item["slot"] == "image_encoder")
    assert slot["binding"]["component_ref"]["id"] == "demo/activation", slot
    assert slot["binding"]["component_ref"]["version"] == "1.2.3", slot
    assert slot["binding"]["params"] == {"negative_slope": 0.25, "tag": "big"}, slot
    assert {item["slot"] for item in assembly["slots"]} == {"image_encoder", "tabular_encoder", "fusion", "head"}, assembly
    generated = project / "generated/model_assembly.py"
    assert generated.is_file(), (output[-2200:], result.stderr[-1000:])
    assert "Generate assembly adapter" in output and "Generated code re-import and forward" in output and '"forward":"passed"' in output.replace("\n", ""), output
    original = generated.read_bytes()
    conflict = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project,
                              env=env, input="\na\ngenerate\n\ny\nq\n", capture_output=True,
                              text=True, timeout=90, check=True)
    assert generated.read_bytes() == original, conflict.stdout
    assert "error" in conflict.stdout.lower(), conflict.stdout
    wrapper = project / "generation-race.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if "generate" in sys.argv and "--if-config-revision" in sys.argv:
    path = Path(".model-maven/project.json")
    config = json.loads(path.read_text())
    config["concurrent_marker"] = "preserve"
    path.write_text(json.dumps(config))
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    raced = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                           env=env, input="\na\ngenerate\ngenerated/raced.py\ny\nq\n", capture_output=True,
                           text=True, timeout=90, check=True)
    assert "PROJECT_CONFIG_CONFLICT" in raced.stdout and not (project / "generated/raced.py").exists(), raced.stdout
    assert json.loads((project / ".model-maven/project.json").read_text())["concurrent_marker"] == "preserve"
    rebuilt = subprocess.run(["python3", "-c", "from generated.model_assembly import build_model; import model_starter; m=build_model(); x=m(**model_starter.batch()['kwargs']); assert tuple(x.shape)==(2,3); assert m.encoders['enc_image'].negative_slope==0.25"], cwd=project, env=env, capture_output=True, text=True, timeout=90)
    assert rebuilt.returncode == 0, (rebuilt.stdout, rebuilt.stderr)
    print("Two encoders, custom fusion and head survive code generation and a fresh Python forward")
    print("TUI discovers an exact component version, validates fields, assembles and probes it")

with tempfile.TemporaryDirectory(prefix="maven-generated-forward-failure-") as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "preferences.json"), COLUMNS="160")
    def command(*args):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120)
        return proc, json.loads(proc.stdout)
    args = ("scaffold", "--input", "x=4", "--output-dim", "2")
    proc, plan = command(*args)
    assert proc.returncode == 0, plan
    proc, report = command(*args, "--apply", "--if-plan", plan["data"]["plan_revision"])
    assert proc.returncode == 0, report
    factory = project / "stateful.py"
    factory.write_text("from torch import nn\ndef encoder(): return nn.Identity()\n")
    proc, report = command("assembly", "add", "x_encoder", "--factory", "stateful:encoder")
    assert proc.returncode == 0, report
    factory.write_text('''from pathlib import Path
from torch import nn
def encoder():
    count = Path("counter.txt")
    value = int(count.read_text()) if count.exists() else 0
    count.write_text(str(value + 1))
    return nn.Identity() if value == 0 else nn.Linear(2, 2)
''')
    output = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, env=env,
        input="\na\ngenerate\n\ny\nq\n", capture_output=True, text=True, timeout=180, check=True).stdout
    assert "FORWARD_FAILED" in output and not (project / "generated/model_assembly.py").exists(), output
    assert (project / "counter.txt").read_text() == "2", output
    print("Stateful factory changes on re-import: generated forward failure does not publish an adapter")

with tempfile.TemporaryDirectory(prefix="maven-generated-no-batch-") as directory:
    project = Path(directory)
    (project / "fixture.py").write_text("from torch import nn\ndef model(): return nn.Linear(4, 2)\n")
    def command(*args):
        proc = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, capture_output=True, text=True, timeout=120)
        return proc, json.loads(proc.stdout)
    proc, report = command("init", "--model", "fixture:model")
    assert proc.returncode == 0, report
    proc, report = command("assembly", "generate")
    assert proc.returncode == 0, report
    assert report["data"]["generated_validation"]["construction"] == "passed" and report["data"]["generated_validation"]["forward"] == "not_run", report
    assert report["data"]["generated_validation"]["weights"] == "not_loaded", report
    print("Without a batch, generated code is reconstructed but no forward or trained weights are claimed")
