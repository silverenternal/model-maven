#!/usr/bin/env python3
"""Exercise conditional TUI commits against the local CLI in isolated projects."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
tui = os.environ.get("MODEL_MAVEN_TUI_BIN", str(root / "target/debug/m-model-tui"))
cli = os.environ.get("MODEL_MAVEN_CLI", str(root.parent / "model-maven/target/debug/m-model"))

cases = [
    ("stable", "", False, False, False, False),
    ("stable", "y", True, True, False, False),
    ("changes_on_commit", "y", False, False, False, False),
    ("changes_on_recheck", "y", True, False, False, False),
    ("stable", "y", True, False, True, False),
    ("stable", "y", True, True, False, True),
]
if os.environ.get("MODEL_MAVEN_GUARDED_SUCCESS_ONLY") == "1":
    cases = [case for case in cases if case == ("stable", "y", True, True, False, False)]
artifact_root = os.environ.get("MODEL_MAVEN_INTEGRATION_ARTIFACTS_DIR")
for case_index, (factory, answer, should_commit, recheck_ok, race, no_version) in enumerate(cases):
    with tempfile.TemporaryDirectory(prefix="maven-guarded-commit-") as directory:
        project = Path(directory)
        (project / "fixture.py").write_text('''import torch
from torch import nn
from pathlib import Path
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 8)
    def forward(self, x): return self.head(x)
def model(): return Model()
def stable(): return torch.ones(2, 48)
def changes_on_commit():
    path = Path("counter.txt")
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    return torch.full((2, 48), 1.0 if count <= 2 else 2.0)
def changes_on_recheck():
    path = Path("counter.txt")
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    return torch.full((2, 48), 1.0 if count <= 3 else 2.0)
''', encoding="utf-8")
        env = os.environ.copy()
        env["MODEL_MAVEN_TUI_PREFERENCES"] = str(project / "preferences.json")
        env["COLUMNS"] = "160"
        cli_path = cli
        if race or no_version:
            wrapper = project / "cli-wrapper.py"
            wrapper.write_text(f'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if {no_version!r} and "--version" in sys.argv: sys.exit(1)
if {race!r} and "probe" in sys.argv and "--if-variant-base" in sys.argv:
    path = Path(".model-maven/variants/working.json")
    variant = json.loads(path.read_text())
    variant["concurrent_test_change"] = True
    path.write_text(json.dumps(variant))
os.execv({cli!r}, [{cli!r}] + sys.argv[1:])
''', encoding="utf-8")
            wrapper.chmod(0o755)
            cli_path = str(wrapper)
        result = subprocess.run(
            [tui, "--line-mode", "--lang", "en", "--cli", cli_path],
            cwd=project, env=env,
            input=f"\n1\nfixture:model\n2\n\nfixture:{factory}\n7\n48\n{answer}\n" + ("0\n" if answer else "") + ("1\nfixture:model\n0\n2\n\nfixture:stable\n7\n48\n\n0\n" if recheck_ok else "") + "q\n",
            capture_output=True, text=True, timeout=150, check=True,
        )
        output = result.stdout
        assert "Commit guarded variant?" in output, output
        if answer:
            assert "Session receipt · historical result" in output, output
            receipt = output.split("Session receipt · historical result", 1)[1]
            assert "Commit and recheck receipt" in receipt, output
            assert "fixture:model" in receipt and f"fixture:{factory}" in receipt, output
            assert "Input fingerprint" in receipt and "Preview before commit · historical" in receipt, output
            expected_version = "unavailable" if no_version else "m-model "
            assert "CLI version queried after operations · executable not pinned  " + expected_version in receipt, receipt
            for label in ("Diagnosis CLI JSON", "Preview CLI JSON", "Commit CLI JSON"):
                assert label in receipt, receipt
            assert "Recheck CLI JSON" in receipt, receipt
            assert '"protocol_version":' in receipt, receipt
            assert "Complete report shown, wrapped for display." in receipt, receipt
            if recheck_ok:
                assert output.count("Session receipt · historical result") == 3, output
                assert "Saved preview from this session" in output.split("Session receipt · historical result")[-1], output
        assert (project / ".model-maven/variants/working.json").exists() == should_commit, output
        if should_commit:
            assert "Guarded recheck · regenerated input" in output, output
            recheck = output.split("Guarded recheck · regenerated input", 1)[1].split("Model-Maven Workbench", 1)[0]
            if recheck_ok:
                assert "Completed" in recheck and "Failed" not in recheck, output
            else:
                code = "VARIANT_BASE_CONFLICT" if race else "INPUT_CONTENT_MISMATCH"
                assert "Failed" in recheck and code in recheck, output
        elif answer:
            assert "INPUT_CONTENT_MISMATCH" in output, output
            assert "Guarded recheck · regenerated input" not in output, output
        else:
            assert "Guarded recheck · regenerated input" not in output, output
        if artifact_root:
            evidence = Path(artifact_root) / f"{case_index:02}-{factory}"
            evidence.mkdir(parents=True, exist_ok=False)
            (evidence / "fixture.py").write_text((project / "fixture.py").read_text(), encoding="utf-8")
            (evidence / "tui-stdout.txt").write_text(result.stdout, encoding="utf-8")
            (evidence / "tui-stderr.txt").write_text(result.stderr, encoding="utf-8")
            variant = project / ".model-maven/variants/working.json"
            if variant.is_file():
                (evidence / "variant.json").write_bytes(variant.read_bytes())
            (evidence / "case.json").write_text(json.dumps({"batch_factory":factory, "confirmed":bool(answer), "committed":should_commit,
                "recheck_passed":recheck_ok, "variant_conflict_wrapper":race, "version_wrapper":no_version}, indent=2) + "\n")
        print(f"{factory}{' + changed variant' if race else (' + version unavailable' if no_version else '')}, {'confirm' if answer else 'cancel'}: {'committed; recheck ' + ('passed' if recheck_ok else 'rejected') if should_commit else 'no variant written'}")
