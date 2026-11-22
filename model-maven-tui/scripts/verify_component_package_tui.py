#!/usr/bin/env python3
"""Build, install and test a project-local component through the TUI."""
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
python = os.environ.get("MODEL_MAVEN_PYTHON", "python3")
with tempfile.TemporaryDirectory(prefix="maven-component-package-") as directory:
    project = Path(directory) / "project"
    project.mkdir()
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / "prefs.json"))
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=180, check=True)
        return json.loads(result.stdout)
    def inventory():
        result = subprocess.run([python, "-c", "import importlib.metadata,json; print(json.dumps(sorted((d.metadata.get('Name',''),d.version) for d in importlib.metadata.distributions())))"],
                                cwd=project, env=env, text=True, capture_output=True, timeout=90, check=True)
        return json.loads(result.stdout)
    plan = run("scaffold", "--input", "x=4", "--output-dim", "2")
    run("scaffold", "--input", "x=4", "--output-dim", "2", "--apply", "--if-plan", plan["data"]["plan_revision"])
    original_config = (project / ".model-maven/project.json").read_bytes()
    before = inventory()
    slow = Path(directory) / "slow-cli.py"
    slow.write_text(f'''#!/usr/bin/env python3
import os, sys, time
if "component" in sys.argv and "install" in sys.argv: time.sleep(2)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
    slow.chmod(0o755)
    master, slave = pty.openpty()
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", str(slow)], cwd=project,
                             env=env, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    pending = bytearray()
    menu = "list, search, select NUMBER"
    def wait(token, timeout=90):
        needle = token.encode(); deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            index = pending.find(needle)
            if index >= 0:
                end = index + len(needle); result = bytes(pending[:end]); del pending[:end]
                return result.decode(errors="replace")
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-4000:]!r}")
            if select.select([master], [], [], .1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError((token, pending[-4000:]))
    def send(value): os.write(master, value.encode())
    def finish(operation):
        for _ in range(300):
            send("refresh\n"); page = wait(menu)
            if f"{operation} · finished" in page:
                assert "Completed" in page, page
                return page
            time.sleep(.1)
        raise AssertionError(f"{operation} did not finish: {page}")
    def scaffold(answer):
        send("scaffold\n"); wait("Component ID"); send("demo/tui-layer\n")
        wait("Extension kind"); send("\n")
        wait("Component version"); send("\n")
        wait("New project-relative output directory"); send("my-extension\n")
        wait("Type apply to continue"); send(answer + "\n"); wait(menu)
    def install(answer):
        send("install\n"); wait("Project-relative extension package directory"); send("my-extension\n")
        wait("Build isolation"); send("no\n")
        wait("Type apply to continue"); send(answer + "\n"); wait(menu)
    try:
        wait("Open workbench"); send("\nz\n"); wait(menu)
        scaffold("no"); assert not (project / "my-extension").exists()
        scaffold("apply"); finish("component_scaffold")
        assert (project / "my-extension/pyproject.toml").is_file()
        install("no"); assert not run("component", "list")["data"]["components"]
        started = time.monotonic(); install("apply")
        assert time.monotonic() - started < 1.5, "install blocked the component page"
        send("b\n"); wait("Model-Maven Workbench"); send("z\n"); wait(menu)
        page = finish("component_install")
        assert "Installed and selected: demo/tui-layer@0.1.0" in page, page
        send("test\n"); wait("Component parameters"); send("\n")
        wait("Optional batch factory"); send("\n")
        wait("Type apply to continue"); send("apply\n"); wait(menu)
        page = finish("component_test")
        assert '"constructs_nn_module":"passed"' in page.replace("\r", "").replace("\n", ""), page
        assert (project / ".model-maven/project.json").read_bytes() == original_config
        assert inventory() == before
        send("b\n"); wait("Model-Maven Workbench")
        send("a\n"); wait("Action: add, set"); send("\n")
        wait("Factory seam name"); send("x_encoder\n")
        wait("Python factory module:callable"); send("component:demo/tui-layer@0.1.0\n")
        wait("Parameters: fields or json"); send("json\n")
        wait("Component parameters as JSON object"); send("{}\n")
        wait("Save assembly? Type y"); send("y\n")
        wait("Assembly saved")
        send("q\n")
        child.wait(timeout=10); assert child.returncode == 0
        assert run("model", "probe")["status"] == "success"
        assert run("assembly", "show", "--config-only")["data"]["slots"][0]["binding"]["component_ref"]["id"] == "demo/tui-layer"
        run("assembly", "remove", "x_encoder")
        assert json.loads((project / ".model-maven/project.json").read_text()) == json.loads(original_config)
        removal = run("component", "uninstall", "demo/tui-layer@0.1.0")
        assert not removal["data"]["requires_force"], removal
        run("component", "uninstall", "demo/tui-layer@0.1.0", "--apply", "--if-plan", removal["data"]["plan_revision"])
        lost = Path(directory) / "lost-cli.py"
        lost.write_text(f'''#!/usr/bin/env python3
import os, subprocess, sys
if "component" in sys.argv and "install" in sys.argv:
    result = subprocess.run([{cli!r}] + sys.argv[1:], capture_output=True)
    if result.returncode == 0: print("invalid receipt"); sys.exit(0)
    sys.stdout.buffer.write(result.stdout); sys.stderr.buffer.write(result.stderr); sys.exit(result.returncode)
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
        lost.chmod(0o755)
        os.close(master)
        master, slave = pty.openpty()
        child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", str(lost)], cwd=project,
                                 env=env, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
        os.close(slave); pending.clear()
        wait("Open workbench"); send("\nz\n"); wait(menu)
        install("apply")
        for _ in range(300):
            send("refresh\n"); page = wait(menu)
            if "component_install · finished" in page: break
            time.sleep(.1)
        else: raise AssertionError("lost-receipt installation did not finish")
        assert "Installation result not verified" in page, page
        assert len(run("component", "list")["data"]["components"]) == 1
        assert (project / ".model-maven/project.json").read_bytes() == original_config and inventory() == before
        send("b\n"); wait("Model-Maven Workbench"); send("q\n")
        child.wait(timeout=10); assert child.returncode == 0
        print("TUI creates, locally installs, tests and assembles a component; cancellation and Python environment preserved")
    finally:
        if child.poll() is None: child.terminate(); child.wait(timeout=10)
        os.close(master)
