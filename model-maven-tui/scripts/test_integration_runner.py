#!/usr/bin/env python3
"""Exercise integration-runner selection, failure and evidence boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/verify_integration.py"


class IntegrationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="maven-runner-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cli = self.root / "cli"
        self.tui = self.root / "tui"
        self.cli.write_bytes(b"cli fixture")
        self.tui.write_bytes(b"tui fixture")

    def arguments(self, output, python=sys.executable):
        return ["--cli", str(self.cli), "--tui", str(self.tui), "--python", python, "--output", str(output)]

    def test_unknown_selection_and_missing_runtime_are_not_success(self):
        output = self.root / "unknown"
        proc = subprocess.run([sys.executable, str(RUNNER), *self.arguments(output), "--case", "unknown"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown cases", proc.stderr)
        self.assertFalse(output.exists())
        false_program = Path("/usr/bin/false")
        if not false_program.exists():
            self.skipTest("requires a failing executable")
        output = self.root / "runtime"
        proc = subprocess.run([sys.executable, str(RUNNER), *self.arguments(output, str(false_program)), "--case", "csv_typed"], capture_output=True, text=True)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(report["status"], "runtime_unavailable")
        self.assertEqual(report["cases"], [])

    def test_repository_identity_covers_dirty_and_untracked_content(self):
        repo = self.root / "repo"
        repo.mkdir()
        def git(*args):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
        git("init", "-q")
        (repo / "tracked.py").write_text("value = 1\n")
        git("add", "tracked.py")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture")
        (repo / "untracked.py").write_text("value = 2\n")
        output = repo / "evidence"
        output.mkdir()
        spec = importlib.util.spec_from_file_location("integration_runner", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        first = runner.repository(repo, output)["content_identity"]
        self.assertEqual(first["status"], "available")
        self.assertEqual(first["untracked_files"], 1)
        (output / "case.log").write_text("runner output is excluded")
        self.assertEqual(first["sha256"], runner.repository(repo, output)["content_identity"]["sha256"])
        (repo / "untracked.py").write_text("value = 3\n")
        second = runner.repository(repo, output)["content_identity"]
        self.assertNotEqual(first["sha256"], second["sha256"])
        (repo / "tracked.py").write_text("value = 4\n")
        self.assertNotEqual(second["sha256"], runner.repository(repo, output)["content_identity"]["sha256"])

    def test_missing_optional_resnet_dependency_is_incomplete_not_pass(self):
        fixture = self.root / "skip.py"
        fixture.write_text("raise SystemExit(77)\n")
        driver = self.root / "driver-skip.py"
        driver.write_text("""import importlib.util, sys
spec = importlib.util.spec_from_file_location('integration_runner', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.CASES = (('classification_resnet18', sys.argv[2], 'fixture'),)
sys.argv = ['runner'] + sys.argv[3:]
raise SystemExit(module.main())
""")
        output = self.root / "skipped"
        proc = subprocess.run([sys.executable, str(driver), str(RUNNER), str(fixture), *self.arguments(output)], capture_output=True, text=True, timeout=30)
        report = json.loads((output / "report.json").read_text())
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["cases"][0]["status"], "skipped")

    @unittest.skipUnless(os.name == "posix", "runner uses Unix process groups")
    def test_selected_case_timeout_and_binary_change_fail_run(self):
        mutation = self.root / "mutate.py"
        mutation.write_text("import os\nwith open(os.environ['MODEL_MAVEN_CLI'], 'ab') as f: f.write(b'changed')\n")
        sleeper = self.root / "sleep.py"
        sleeper.write_text("import time\ntime.sleep(30)\n")
        output = self.root / "selected"
        driver = self.root / "driver.py"
        driver.write_text("""import importlib.util, sys
spec = importlib.util.spec_from_file_location('integration_runner', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.CASES = (('mutate', sys.argv[2], 'fixture'), ('timeout', sys.argv[3], 'fixture'), ('unselected', sys.argv[2], 'fixture'))
sys.argv = ['runner'] + sys.argv[4:]
raise SystemExit(module.main())
""")
        proc = subprocess.run([sys.executable, str(driver), str(RUNNER), str(mutation), str(sleeper),
                               *self.arguments(output), "--case", "mutate", "--case", "timeout", "--case-timeout", "1"],
                              capture_output=True, text=True, timeout=30)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["selection"]["full_suite"])
        self.assertFalse(report["binaries_unchanged"])
        self.assertEqual([(case["id"], case["exit_code"]) for case in report["cases"]], [("mutate", 0), ("timeout", None)])
        self.assertIn("timed out", (output / "timeout.log").read_text())


if __name__ == "__main__":
    unittest.main()
