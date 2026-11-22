#!/usr/bin/env python3
"""Preview and conditionally uninstall a real registered component in TUI."""
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-component-remove-") as directory:
    project = Path(directory) / "project"
    project.mkdir()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=90, check=True)
        return json.loads(result.stdout)
    plan = run("scaffold", "--input", "x=4", "--output-dim", "2")
    run("scaffold", "--input", "x=4", "--output-dim", "2", "--apply", "--if-plan", plan["data"]["plan_revision"])
    (project / "fixture.py").write_text("from torch import nn\ndef layer(): return nn.ReLU()\n")
    run("component", "add", "demo/layer", "--kind", "layer", "--entrypoint", "fixture:layer")
    reference = "demo/layer@0.1.0"
    preview = run("component", "uninstall", reference)
    manifest = project / preview["data"]["manifest_path"]
    assert len(preview["data"]["plan_revision"]) == 64
    wrapper = Path(directory) / "race-cli.py"
    wrapper.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if "uninstall" in sys.argv and "--apply" in sys.argv:
    with Path({str(manifest)!r}).open("a") as stream: stream.write(" ")
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    wrapper.chmod(0o755)
    slow = Path(directory) / "slow-cli.py"
    slow.write_text(f'''#!/usr/bin/env python3
import os, sys, time
if "uninstall" in sys.argv: time.sleep(2)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    slow.chmod(0o755)

    def interaction(binary, answer, expected, force=False, leave=False):
        master, slave = pty.openpty()
        child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", str(binary)],
                                 cwd=project, env=env, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
        os.close(slave)
        pending = bytearray()
        menu = "list, search, select NUMBER"
        def wait(token):
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                pos = pending.find(token.encode())
                if pos >= 0:
                    end = pos + len(token.encode()); result = bytes(pending[:end]); del pending[:end]
                    return result.decode(errors="replace")
                if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-3000:]!r}")
                if select.select([master], [], [], .1)[0]: pending.extend(os.read(master, 65536))
            raise AssertionError((token, pending[-4000:]))
        def send(s): os.write(master, s.encode())
        try:
            wait("Open workbench"); send("\nz\n"); wait(menu)
            send("list\n"); wait(menu)
            for _ in range(100):
                send("refresh\n"); page = wait(menu)
                if "1. component_list · finished" in page: break
                time.sleep(.1)
            assert f"1. {reference}" in page, page
            send("select 1\n"); wait(menu)
            started = time.monotonic()
            send("uninstall\n"); wait(menu)
            if leave:
                assert time.monotonic() - started < 1.5, "preview blocked the interface"
                send("b\n"); wait("Model-Maven Workbench"); send("z\n"); wait(menu)
            for _ in range(100):
                send("refresh\n"); page = wait(menu)
                if "component_uninstall_preview · finished" in page: break
                time.sleep(.1)
            else: raise AssertionError("removal preview did not complete")
            send("review\n"); wait("Component removal preview")
            wait("Type FORCE to accept the references and remove" if force else "Type the exact component ID@version to remove")
            started = time.monotonic()
            send(answer + "\n")
            wait(menu)
            if leave and expected:
                assert time.monotonic() - started < 1.5, "removal blocked the interface"
                send("b\n"); wait("Model-Maven Workbench"); send("z\n"); wait(menu)
            if expected:
                for _ in range(100):
                    send("refresh\n"); page = wait(menu)
                    if "component_uninstall_apply · finished" in page: break
                    time.sleep(.1)
                else: raise AssertionError("removal did not complete")
                assert expected in page, page
            elif answer == "no":
                send("review\n"); wait("No unreviewed removal plan"); wait(menu)
            send("b\n"); wait("Model-Maven Workbench"); send("q\n")
            child.wait(timeout=10); assert child.returncode == 0
        finally:
            if child.poll() is None: child.terminate(); child.wait(timeout=10)
            os.close(master)

    interaction(slow, "no", None, leave=True)
    assert manifest.exists()
    interaction(wrapper, reference, "COMPONENT_PLAN_CONFLICT")
    assert manifest.exists()
    run("assembly", "add", "x_encoder", "--component", reference)
    assert run("component", "uninstall", reference)["data"]["requires_force"] is True
    interaction(cli, "no", None, force=True)
    assert manifest.exists()
    interaction(slow, "FORCE", "Completed", force=True, leave=True)
    assert not manifest.exists() and (project / "fixture.py").is_file()
    assert "demo/layer" in (project / ".model-maven/project.json").read_text()
    run("assembly", "remove", "x_encoder")
    run("component", "add", "demo/layer", "--kind", "layer", "--entrypoint", "fixture:layer")
    lost = Path(directory) / "lost-cli.py"
    lost.write_text(f'''#!/usr/bin/env python3
import os, subprocess, sys
if "uninstall" in sys.argv and "--apply" in sys.argv:
    result = subprocess.run([{cli!r}] + sys.argv[1:], capture_output=True)
    if result.returncode == 0: print("invalid receipt"); sys.exit(0)
    sys.stdout.buffer.write(result.stdout); sys.stderr.buffer.write(result.stderr); sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    lost.chmod(0o755)
    interaction(lost, reference, "Removal result not verified")
    assert not manifest.exists() and (project / "fixture.py").is_file()
    run("component", "scaffold", "demo/wheel", "--kind", "layer", "--output", "extension-src")
    run("component", "install", "extension-src", "--no-build-isolation")
    reference = "demo/wheel@0.1.0"
    wheel_plan = run("component", "uninstall", reference)["data"]
    assert wheel_plan["wheel_removed"] is True and wheel_plan["removal_tree"]["files"] > 0, wheel_plan
    extension_dir = project / wheel_plan["extension_path"]
    added = extension_dir / "changed-after-preview.txt"
    tree_race = Path(directory) / "tree-race.py"
    tree_race.write_text(f'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if "uninstall" in sys.argv and "--apply" in sys.argv:
    Path({str(added)!r}).write_text("retain user changes")
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    tree_race.chmod(0o755)
    interaction(tree_race, reference, "COMPONENT_PLAN_CONFLICT")
    assert added.is_file() and extension_dir.is_dir()
    added.unlink()
    interaction(cli, reference, "Completed")
    assert not extension_dir.exists()
    print("TUI previews and cancels, rejects changed plan, then removes the component without deleting source")
