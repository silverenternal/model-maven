#!/usr/bin/env python3
"""Exercise component discovery, inspection and explicit conformance testing."""
import csv
import fcntl
import json
import os
import pty
import select
import shutil
import struct
import subprocess
import tempfile
import termios
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
backend = root.parent / "model-maven"
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(backend / "target/debug/m-model"))
with tempfile.TemporaryDirectory(prefix="maven-components-pair-") as directory:
    project = Path(directory)
    (project / "fixture.py").write_text('''from pathlib import Path
import torch
from torch import nn
def model(): return nn.Linear(4, 2)
def batch(): return torch.ones(2, 4)
def layer():
    Path("factory-ran").write_text("called")
    return nn.ReLU()
''')
    def run(*args):
        result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, check=True, capture_output=True, text=True)
        return json.loads(result.stdout)
    run("init", "--model", "fixture:model", "--batch", "fixture:batch")
    run("component", "add", "demo/activation", "--kind", "layer", "--entrypoint", "fixture:layer")
    site = project / "site"
    site.mkdir()
    source = backend / "examples/extensions/silu-component/src/model_maven_silu"
    shutil.copytree(source, site / "model_maven_silu")
    info = site / "model_maven_silu_example-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: model-maven-silu-example\nVersion: 0.1.0\n\n")
    (info / "entry_points.txt").write_text("[model_maven.components]\nsilu = model_maven_silu.layer:make_silu\n")
    with (info / "RECORD").open("w", newline="") as stream:
        writer = csv.writer(stream)
        for file in site.rglob("*"):
            if file.is_file(): writer.writerow([file.relative_to(site).as_posix(), "", ""])
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 110, 0, 0))
    env = os.environ.copy()
    env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "prefs.json")
    child = subprocess.Popen([tui, "--line-mode", "--lang", "en", "--cli", cli], cwd=project, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    pending = bytearray()
    menu = "list, search, select NUMBER"
    def until(token, timeout=60):
        needle, deadline = token.encode(), time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos = pending.find(needle)
            if pos >= 0:
                end = pos + len(needle)
                result = bytes(pending[:end])
                del pending[:end]
                return result.decode(errors="replace")
            if child.poll() is not None: raise AssertionError(f"TUI exited: {pending[-5000:]!r}")
            if select.select([master], [], [], 0.1)[0]: pending.extend(os.read(master, 65536))
        raise AssertionError(f"missing {token!r}: {pending[-5000:]!r}")
    def send(value): os.write(master, value.encode())
    def finish(number, operation):
        for _ in range(100):
            send("refresh\n")
            page = until(menu)
            if f"{number}. {operation} · finished" in page:
                section = page.split(f"{number}. {operation} · finished", 1)[1]
                assert "Completed" in section, page
                return page
            time.sleep(0.3)
        raise AssertionError(f"{operation} did not finish: {page}")
    def discover(apply):
        send("discover\n")
        until("Distribution filter")
        send("model-maven-silu-example\n")
        until("Optional additional package directory")
        send(str(site) + "\n")
        until("Type apply to continue")
        send(("apply" if apply else "n") + "\n")
        return until(menu)
    def component_test(apply):
        send("test\n")
        until("Component parameters")
        send("\n")
        until("Optional batch factory")
        send("\n")
        until("Type apply to continue")
        send(("apply" if apply else "n") + "\n")
        return until(menu)
    try:
        until("Open workbench")
        send("\nz\n")
        until(menu)
        discover(False)
        assert len(run("component", "list")["data"]["components"]) == 1
        discover(True)
        page = finish(1, "component_discover")
        assert "example/silu" in page and not (project / "factory-ran").exists()
        send("list\n")
        until(menu)
        page = finish(2, "component_list")
        assert "1. demo/activation@0.1.0" in page and "2. example/silu@0.1.0" in page
        send("select 1\n")
        page = until(menu)
        assert "Selected: demo/activation@0.1.0" in page
        send("show\n")
        until(menu)
        page = finish(3, "component_show")
        assert '"params_schema"' in page and '"not_run"' in page
        send("check\n")
        until(menu)
        page = finish(4, "component_check")
        assert '"dependency_resolution":"passed"' in page.replace("\r", "").replace("\n", "") and not (project / "factory-ran").exists(), page
        component_test(False)
        assert not (project / "factory-ran").exists()
        component_test(True)
        page = finish(5, "component_test")
        assert '"constructs_nn_module":"passed"' in page.replace("\r", "").replace("\n", "") and (project / "factory-ran").exists(), page
        for index in range(25):
            run("component", "add", f"zz/bulk{index:02}", "--kind", "layer", "--entrypoint", "fixture:layer")
        send("list\n")
        until(menu)
        page = finish(6, "component_list")
        assert "1/2" in page and "20. zz/bulk17@0.1.0" in page and "21. zz/bulk18@0.1.0" not in page, page
        send("n\n")
        page = until(menu)
        assert "2/2" in page and "21. zz/bulk18@0.1.0" in page, page
        send("select 21\n")
        page = until(menu)
        assert "Selected: zz/bulk18@0.1.0" in page, page
        send("select 1\n")
        until("Select a number shown on the current page.")
        page = until(menu)
        assert "Selected: zz/bulk18@0.1.0" in page and "2/2" in page, page
        send("show\n")
        until(menu)
        page = finish(7, "component_show")
        assert '"component_id":"zz/bulk18"' in page.replace("\r", "").replace("\n", ""), page
        send("p\n")
        page = until(menu)
        assert "1/2" in page and "1. demo/activation@0.1.0" in page, page
        print("Real component list paginates and selects a component beyond the first page")
        (project / "factory-ran").unlink()
        def add_component(confirm, advanced=False):
            send("add\n")
            until("Component ID")
            send("local/custom\n")
            until("Kind:")
            send("\n")
            until("Python factory")
            send("fixture:layer\n")
            until("Component version")
            send("\n")
            until("Description (optional)")
            send("User activation\n")
            until("Configure schema")
            send(("y" if advanced else "n") + "\n")
            if advanced:
                until("Parameter schema JSON")
                send('{"type":"object","properties":{},"required":[],"additionalProperties":false}\n')
                until("Capabilities JSON")
                send("\n")
                until("Dependencies JSON")
                send("[]\n")
                until("Conformance fixture")
                send("\n")
            until("Type apply to continue")
            send(("apply" if confirm else "n") + "\n")
            return until(menu)
        before = len(run("component", "list")["data"]["components"])
        add_component(False)
        assert len(run("component", "list")["data"]["components"]) == before
        add_component(True, advanced=True)
        page = finish(8, "component_add")
        assert "Registered and selected: local/custom@0.1.0" in page, page
        item = run("component", "show", "local/custom@0.1.0")["data"]["component"]["manifest"]
        assert item["entrypoint"] == "fixture:layer" and item["params_schema"]["additionalProperties"] is False, item
        assert not (project / "factory-ran").exists()
        add_component(True)
        for _ in range(100):
            send("refresh\n")
            page = until(menu)
            if "component_add · finished" in page and "COMPONENT_REGISTER_FAILED" in page:
                break
            time.sleep(0.3)
        else:
            raise AssertionError(f"duplicate registration did not fail: {page}")
        assert len(run("component", "list")["data"]["components"]) == before + 1
        assert not (project / "factory-ran").exists()
        print("TUI registers an existing factory with schema, preserves cancellation and rejects duplicates without running code")
        send("b\n")
        until("Model-Maven Workbench")
        send("q\n")
        child.wait(timeout=10)
        assert child.returncode == 0
        print("Static package discovery, versioned selection, metadata/schema, dependency check, cancelled and executed component test verified")
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
