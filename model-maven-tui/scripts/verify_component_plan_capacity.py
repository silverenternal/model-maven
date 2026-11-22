#!/usr/bin/env python3
"""Eight pending removal plans must not prevent reviewing and cancelling one."""
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
with tempfile.TemporaryDirectory(prefix="maven-component-capacity-") as directory:
    base = Path(directory)
    projects = []
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(base / "prefs.json"))
    def run(project, *args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project,
                                env=env, capture_output=True, text=True, timeout=90, check=True)
        return json.loads(result.stdout)
    for index in range(8):
        project = base / f"project-{index}"
        project.mkdir(); projects.append(project)
        plan = run(project, "scaffold", "--input", "x=4", "--output-dim", "2")
        run(project, "scaffold", "--input", "x=4", "--output-dim", "2", "--apply", "--if-plan", plan["data"]["plan_revision"])
        (project / "fixture.py").write_text("from torch import nn\ndef layer(): return nn.ReLU()\n")
        run(project, "component", "add", "demo/layer", "--kind", "layer", "--entrypoint", "fixture:layer")
    master, slave = pty.openpty()
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=projects[0],
                             env=env, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave); pending = bytearray(); menu = "list, search, select NUMBER"
    def wait(token, timeout=60):
        needle = token.encode(); deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            position = pending.find(needle)
            if position >= 0:
                end = position + len(needle); output = bytes(pending[:end]).decode(errors="replace"); del pending[:end]
                return output
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-4000:]!r}")
            if select.select([master], [], [], .1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError((token, pending[-4000:]))
    def send(value): os.write(master, value.encode())
    def refresh_until(token):
        for _ in range(150):
            send("refresh\n"); page = wait(menu)
            if token in page: return page
            time.sleep(.1)
        raise AssertionError((token, page))
    try:
        wait("Open workbench"); send("\n"); wait("Model-Maven Workbench")
        for index, project in enumerate(projects):
            if index:
                send("o\n"); wait("Project directory"); send(str(project) + "\n"); wait("Model-Maven Workbench")
            send("z\n"); wait(menu)
            send("list\n"); wait(menu); refresh_until("component_list · finished")
            send("select 1\n"); wait(menu)
            send("uninstall\n"); wait(menu); refresh_until("component_uninstall_preview · finished")
            if index < 7:
                send("b\n"); wait("Model-Maven Workbench")
        send("review\n"); wait("Component removal preview"); wait("Type the exact component ID@version to remove")
        send("no\n"); wait(menu)
        send("review\n"); wait("No unreviewed removal plan"); wait(menu)
        send("list\n"); wait(menu); refresh_until("component_list · finished")
        assert len(run(projects[-1], "component", "list")["data"]["components"]) == 1
        send("select 1\n"); wait(menu)
        send("uninstall\n"); wait(menu); refresh_until("component_uninstall_preview · finished")
        send("review\n"); wait("Component removal preview"); wait("Type the exact component ID@version to remove")
        send("demo/layer@0.1.0\n"); wait(menu); refresh_until("component_uninstall_apply · finished")
        assert not run(projects[-1], "component", "list")["data"]["components"]
        assert all(len(run(project, "component", "list")["data"]["components"]) == 1 for project in projects[:-1])
        send("b\n"); wait("Model-Maven Workbench"); send("q\n")
        child.wait(timeout=10); assert child.returncode == 0
        print("Eight project-scoped previews: review, cancel and confirmed removal work without affecting other projects")
    finally:
        if child.poll() is None: child.terminate(); child.wait(timeout=10)
        os.close(master)
