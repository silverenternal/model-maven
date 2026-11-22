#!/usr/bin/env python3
"""Inject concurrent variant, batch or source changes before a TUI layer commit."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))
for race in ("variant", "input", "source"):
    with tempfile.TemporaryDirectory(prefix="maven-layer-race-") as directory:
        project = Path(directory)
        env = os.environ.copy()
        env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
        (project / "fixture.py").write_text('''import torch
from torch import nn
from pathlib import Path
def model(): return nn.Sequential(nn.Linear(4, 4), nn.ReLU())
def batch(): return torch.full((2, 4), float(Path("value.txt").read_text()))
''')
        (project / "value.txt").write_text("1")
        def run(*args):
            result = subprocess.run([cli, "--json", "--no-interactive", *args], cwd=project, env=env,
                                    capture_output=True, text=True, timeout=120)
            report = json.loads(result.stdout)
            assert result.returncode == 0, report
            return report
        run("init", "--model", "fixture:model", "--batch", "fixture:batch")
        if race != "source": run("layer", "set", "model.0", "--param", "bias=true")
        variant_path = project / ".model-maven/variants/working.json"
        before = variant_path.read_bytes() if variant_path.exists() else None
        modules = run("layer", "list", *( ["--variant", "working"] if race != "source" else []))["data"]["modules"]
        index = next(i for i, item in enumerate(modules, 1) if item["path"] == "model.1")
        wrapper = project / "race-cli.py"
        wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if "layer" in sys.argv and "--if-variant-base" in sys.argv:
    if {race!r} == "input":
        Path("value.txt").write_text("2")
    elif {race!r} == "source":
        path = Path("fixture.py")
        path.write_text(path.read_text() + "\\n# concurrent source edit\\n")
    else:
        path = Path(".model-maven/variants/working.json")
        value = json.loads(path.read_text())
        value["concurrent_marker"] = "preserve"
        path.write_text(json.dumps(value))
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''')
        wrapper.chmod(0o755)
        result = subprocess.run([tui, "--line-mode", "--lang", "en", "--cli", str(wrapper)], cwd=project,
                                env=env, input=f"\nw\n{index}\nreplace\nGELU\n{{}}\ny\nq\n",
                                capture_output=True, text=True, timeout=120, check=True)
        code = {"input": "INPUT_CONTENT_MISMATCH", "variant": "VARIANT_BASE_CONFLICT", "source": "SOURCE_MANIFEST_CONFLICT"}[race]
        assert code in result.stdout, result.stdout
        assert "independent recheck" not in result.stdout, result.stdout
        if race == "input": assert variant_path.read_bytes() == before
        elif race == "variant": assert json.loads(variant_path.read_text())["concurrent_marker"] == "preserve"
        else: assert not variant_path.exists() and "# concurrent source edit" in (project / "fixture.py").read_text()
        print(f"{race} changed after preview: guarded commit refused")
