#!/usr/bin/env python3
"""Offline acceptance for Model-Maven's project environment lifecycle."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import tempfile
import time
import zipfile


def verify_workbench_environment_setup(binary, project, environment):
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(binary)],
        cwd=project,
        env=environment,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    transcript = bytearray()
    deadline = time.monotonic() + 60
    try:
        os.write(master, b"5\n2\n1\n\nback\n8\n")
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                raise RuntimeError("workbench environment setup PTY timed out")
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    transcript.extend(os.read(master, 4096))
                except OSError as error:
                    if error.errno == 5:  # Linux PTY returns EIO after the child closes.
                        break
                    raise
        process.wait(timeout=2)
        while select.select([master], [], [], 0)[0]:
            try:
                transcript.extend(os.read(master, 4096))
            except OSError as error:
                if error.errno == 5:
                    break
                raise
        rendered = transcript.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise RuntimeError(f"workbench environment setup failed:\n{rendered}")
        if "项目虚拟环境已创建并验证" not in rendered:
            raise RuntimeError(f"workbench did not confirm environment creation:\n{rendered}")
        if "/.venv/bin/python" not in rendered:
            raise RuntimeError(f"workbench did not identify the created project interpreter:\n{rendered}")
        if "本次工作台会话已切换到项目虚拟环境" not in rendered:
            raise RuntimeError(f"workbench did not select the newly created interpreter:\n{rendered}")
        return "environment setup page created and selected the project venv in a PTY"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


def build_wheel(directory, distribution, version, package, module_source):
    normalized = distribution.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    contents = {
        f"{package}/__init__.py": module_source.encode(),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: model-maven-offline-acceptance\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode(),
    }
    rows = []
    for name, content in contents.items():
        rows.append([name, f"sha256={base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=')}", str(len(content))])
    rows.append([f"{dist_info}/RECORD", "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    contents[f"{dist_info}/RECORD"] = record.getvalue().encode()
    wheel = directory / f"{normalized}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    return wheel


def invoke(binary, project, environment, *arguments, python=None, expect_success=True):
    command = [str(binary)]
    if python:
        command.extend(["--python", str(python)])
    command.extend(["--json", *arguments])
    result = subprocess.run(
        command,
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"expected one JSON report from {command!r}; stdout={result.stdout!r}; "
            f"stderr={result.stderr!r}"
        ) from error
    if expect_success and result.returncode != 0:
        raise RuntimeError(f"command failed: {command!r}\n{report!r}\n{result.stderr}")
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f"command unexpectedly succeeded: {command!r}\n{report!r}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/m-model"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)

    with tempfile.TemporaryDirectory(prefix="model-maven-env-acceptance-") as temporary:
        root = Path(temporary)
        project = root / "project"
        wheelhouse = root / "wheelhouse"
        project.mkdir()
        wheelhouse.mkdir()
        torch_wheel = build_wheel(wheelhouse, "torch", "0.1.0", "torch", "__version__ = '0.1.0'\n")
        dependency_wheel = build_wheel(wheelhouse, "sample-dep", "1.0.0", "sample_dep", "VALUE = 1\n")
        extra_wheel = build_wheel(wheelhouse, "extra-dep", "1.0.0", "extra_dep", "VALUE = 2\n")
        requirements = project / "requirements.txt"
        nested_requirements = project / "requirements/base.txt"
        nested_requirements.parent.mkdir()
        nested_requirements.write_text("torch==0.1.0\nsample-dep==1.0.0\n")
        requirements.write_text("-r requirements/base.txt\n")

        environment = os.environ.copy()
        environment.update({
            "PIP_NO_INDEX": "1",
            "PIP_FIND_LINKS": str(wheelhouse),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
        })

        locked = invoke(
            binary, project, environment,
            "env", "lock", "--requirements", "requirements.txt",
            python=args.python,
        )
        lock_text = (project / "requirements.lock.txt").read_text()
        lock_metadata = json.loads((project / ".model-maven/environment.lock.json").read_text())
        assert locked["data"]["package_count"] == 2
        assert lock_metadata["requirements_inputs"] == ["requirements.txt", "requirements/base.txt"]
        assert f"sha256:{hashlib.sha256(torch_wheel.read_bytes()).hexdigest()}" in lock_text
        assert f"sha256:{hashlib.sha256(dependency_wheel.read_bytes()).hexdigest()}" in lock_text
        assert str(wheelhouse) not in json.dumps(lock_metadata)

        mismatched_python = root / "python-with-different-abi"
        mismatched_python.write_text(
            f"#!{sys.executable}\n"
            "import json, subprocess, sys\n"
            f"real_python = {sys.executable!r}\n"
            "arguments = sys.argv[1:]\n"
            "if arguments[:1] == ['-c'] and 'cache_tag' in arguments[1]:\n"
            "    result = subprocess.run([real_python, *arguments], check=True, capture_output=True, text=True)\n"
            "    identity = json.loads(result.stdout)\n"
            "    identity['soabi'] = str(identity['soabi']) + '-different'\n"
            "    print(json.dumps(identity))\n"
            "else:\n"
            "    import os\n"
            "    os.execv(real_python, [real_python, *arguments])\n"
        )
        mismatched_python.chmod(0o755)
        abi_drift = invoke(
            binary, project, environment,
            "env", "lock", "--requirements", "requirements.txt",
            python=str(mismatched_python), expect_success=False,
        )
        assert abi_drift["diagnostics"][0]["code"] == "ENVIRONMENT_LOCK_STALE"

        invoke(
            binary,
            project,
            environment,
            "init",
            "--model",
            "fixture:build_model",
            python=args.python,
        )
        workbench_environment_result = verify_workbench_environment_setup(
            binary, project, environment
        )
        assert (project / ".venv/bin/python").is_file()
        setup = invoke(
            binary, project, environment,
            "env", "setup", "--requirements", "requirements.txt",
            python=args.python,
        )
        assert setup["data"]["created"] is False
        assert setup["data"]["environment_check"]["matches"] is True

        checked = invoke(binary, project, environment, "env", "check")
        assert checked["data"]["environment_check"]["matches"] is True

        custom = invoke(
            binary, project, environment,
            "env", "setup", "--requirements", "requirements.txt", "--venv", ".env-custom",
            python=args.python,
        )
        assert custom["data"]["created"] is True
        assert json.loads((project / ".model-maven/environment.json").read_text())["venv"] == ".env-custom"

        selected = invoke(binary, project, environment, "env", "check")
        assert selected["data"]["environment_check"]["venv"] == ".env-custom"
        doctor = invoke(binary, project, environment, "doctor")
        assert Path(doctor["data"]["python"]).parent.name == "bin"
        assert ".env-custom" in doctor["data"]["python"]
        assert doctor["data"]["torch_available"] is True

        reused = invoke(binary, project, environment, "env", "setup")
        assert reused["data"]["created"] is False

        custom_python = project / ".env-custom/bin/python"
        subprocess.run(
            [str(custom_python), "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), "extra-dep==1.0.0"],
            cwd=project,
            env=environment,
            check=True,
            timeout=120,
            stdout=subprocess.DEVNULL,
        )
        drift = invoke(binary, project, environment, "env", "check", expect_success=False)
        assert drift["diagnostics"][0]["code"] == "ENVIRONMENT_DRIFT"
        assert "extra-dep" in drift["data"]["extra"]

        requirements.write_text("torch==0.1.0\nsample-dep==1.0.0\n# changed input\n")
        input_drift = invoke(binary, project, environment, "env", "check", expect_success=False)
        assert input_drift["diagnostics"][0]["code"] == "ENVIRONMENT_DRIFT"
        assert input_drift["data"]["requirements_match"] is False
        stale = invoke(
            binary, project, environment,
            "env", "setup",
            expect_success=False,
        )
        assert stale["diagnostics"][0]["code"] == "ENVIRONMENT_LOCK_STALE"

        unsafe_project = root / "unsafe-project"
        unsafe_project.mkdir()
        (unsafe_project / "requirements.txt").write_text(
            "torch @ https://user:secret@example.invalid/torch-0.1.0.whl\n"
        )
        unsafe = invoke(
            binary, unsafe_project, environment,
            "env", "lock", expect_success=False, python=args.python,
        )
        assert unsafe["diagnostics"][0]["code"] == "ENVIRONMENT_SOURCE_DEPENDENCY_UNSUPPORTED"
        assert "user" not in json.dumps(unsafe) and "secret" not in json.dumps(unsafe)
        unsafe_requirements = unsafe_project / "requirements.txt"
        for unsupported_source in [
            "git+https://example.invalid/team/package.git\n",
            "-e .\n",
            "sample-dep @ file:///tmp/sample-dep.whl\n",
        ]:
            unsafe_requirements.write_text(unsupported_source)
            rejected_source = invoke(
                binary, unsafe_project, environment,
                "env", "lock", "--requirements", "requirements.txt",
                python=args.python, expect_success=False,
            )
            assert rejected_source["diagnostics"][0]["code"] == "ENVIRONMENT_SOURCE_DEPENDENCY_UNSUPPORTED"

        failed_project = root / "failed-project"
        failed_project.mkdir()
        (failed_project / "requirements.txt").write_text("torch==0.1.0\nsample-dep==1.0.0\n")
        existing_environment = failed_project / ".existing-env"
        subprocess.run([args.python, "-m", "venv", str(existing_environment)], check=True, timeout=120)
        existing_selection = {
            "schema_version": 1,
            "venv": ".existing-env",
        }
        (failed_project / ".model-maven").mkdir()
        (failed_project / ".model-maven/environment.json").write_text(
            json.dumps(existing_selection, indent=2) + "\n"
        )
        failing_python = root / "python-that-fails-venv-creation"
        failing_python.write_text(
            f"#!{sys.executable}\n"
            "import os, subprocess, sys\n"
            f"real_python = {sys.executable!r}\n"
            "arguments = sys.argv[1:]\n"
            "if arguments[:2] == ['-m', 'venv']:\n"
            "    subprocess.run([real_python, *arguments], check=True)\n"
            "    print('simulated failure after partial environment creation', file=sys.stderr)\n"
            "    raise SystemExit(23)\n"
            "os.execv(real_python, [real_python, *arguments])\n"
        )
        failing_python.chmod(0o755)
        invoke(
            binary, failed_project, environment,
            "env", "lock", "--requirements", "requirements.txt",
            python=str(failing_python),
        )
        failed_setup = invoke(
            binary, failed_project, environment,
            "env", "setup", "--requirements", "requirements.txt", "--venv", ".env-failed",
            python=str(failing_python), expect_success=False,
        )
        assert failed_setup["diagnostics"][0]["code"] == "ENVIRONMENT_PROCESS_FAILED"
        assert not (failed_project / ".env-failed").exists()
        assert (existing_environment / "bin/python").is_file()
        assert json.loads((failed_project / ".model-maven/environment.json").read_text()) == existing_selection

        install_failing_python = root / "python-with-failing-install"
        install_failing_python.write_text(
            f"#!{sys.executable}\n"
            "import os, subprocess, sys\n"
            f"real_python = {sys.executable!r}\n"
            "arguments = sys.argv[1:]\n"
            "if arguments[:2] == ['-m', 'venv']:\n"
            "    subprocess.run([real_python, *arguments], check=True)\n"
            "    target = arguments[2]\n"
            "    target_python = os.path.join(target, 'Scripts/python.exe' if os.name == 'nt' else 'bin/python')\n"
            "    pip_main = subprocess.check_output([target_python, '-c', 'import pip,pathlib; print(pathlib.Path(pip.__file__).parent / \\\"__main__.py\\\")'], text=True).strip()\n"
            "    with open(pip_main, 'w', encoding='utf-8') as stream:\n"
            "        stream.write('raise SystemExit(23)\\n')\n"
            "    raise SystemExit(0)\n"
            "os.execv(real_python, [real_python, *arguments])\n"
        )
        install_failing_python.chmod(0o755)
        failed_install = invoke(
            binary, failed_project, environment,
            "env", "setup", "--requirements", "requirements.txt", "--venv", ".env-install-failed",
            python=str(install_failing_python), expect_success=False,
        )
        assert failed_install["diagnostics"][0]["code"] == "ENVIRONMENT_PROCESS_FAILED"
        assert not (failed_project / ".env-install-failed").exists()
        assert (existing_environment / "bin/python").is_file()
        assert json.loads((failed_project / ".model-maven/environment.json").read_text()) == existing_selection

        non_torch_project = root / "non-torch-project"
        non_torch_project.mkdir()
        (non_torch_project / "requirements.txt").write_text("sample-dep==1.0.0\n")
        generic_lock = invoke(
            binary, non_torch_project, environment,
            "env", "lock", "--requirements", "requirements.txt",
            python=args.python,
        )
        generic_metadata = json.loads(
            (non_torch_project / ".model-maven/environment.lock.json").read_text()
        )
        assert generic_lock["data"]["package_count"] == 1
        assert [item["name"] for item in generic_metadata["packages"]] == ["sample-dep"]
        generic_setup = invoke(
            binary, non_torch_project, environment,
            "env", "setup", "--requirements", "requirements.txt",
            python=args.python,
        )
        assert generic_setup["data"]["environment_check"]["matches"] is True
        generic_doctor = invoke(binary, non_torch_project, environment, "doctor")
        assert generic_doctor["data"]["torch_available"] is False
        generic_model = invoke(
            binary, non_torch_project, environment,
            "model", "inspect", "--model", "example:build_model", expect_success=False,
        )
        assert generic_model["diagnostics"][0]["code"] == "OPERATION_CAPABILITY_UNAVAILABLE"
        assert "PyTorch" in generic_model["data"]["readiness"]["readiness"]["capabilities"][0]["reason"]

        cancel_marker = root / "cancel-venv-created"
        cancelling_python = root / "python-waiting-during-venv-setup"
        cancelling_python.write_text(
            f"#!{sys.executable}\n"
            "import os, subprocess, sys, time\n"
            f"real_python = {sys.executable!r}\n"
            f"marker = {str(cancel_marker)!r}\n"
            "arguments = sys.argv[1:]\n"
            "if arguments[:2] == ['-m', 'venv']:\n"
            "    subprocess.run([real_python, *arguments], check=True)\n"
            "    Path = __import__('pathlib').Path\n"
            "    Path(marker).write_text('ready')\n"
            "    time.sleep(30)\n"
            "    raise SystemExit(24)\n"
            "os.execv(real_python, [real_python, *arguments])\n"
        )
        cancelling_python.chmod(0o755)
        cancelled = subprocess.Popen(
            [
                str(binary), "--json", "--python", str(cancelling_python),
                "env", "setup", "--requirements", "requirements.txt", "--venv", ".env-cancelled",
            ],
            cwd=failed_project,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cancellation_deadline = time.monotonic() + 15
        while not cancel_marker.exists() and cancelled.poll() is None and time.monotonic() < cancellation_deadline:
            time.sleep(0.05)
        if not cancel_marker.exists():
            cancelled.kill()
            stdout, stderr = cancelled.communicate(timeout=5)
            raise RuntimeError(
                f"environment setup did not reach the cancellable venv phase; "
                f"status={cancelled.returncode}; stdout={stdout!r}; stderr={stderr!r}"
            )
        cancelled.send_signal(signal.SIGINT)
        stdout, stderr = cancelled.communicate(timeout=10)
        try:
            cancellation_report = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"cancellation returned no JSON report: {stdout!r}; stderr={stderr!r}") from error
        assert cancelled.returncode == 130
        assert cancellation_report["diagnostics"][0]["code"] == "USER_INTERRUPTED"
        assert not (failed_project / ".env-cancelled").exists()
        assert (existing_environment / "bin/python").is_file()
        assert json.loads((failed_project / ".model-maven/environment.json").read_text()) == existing_selection

    print(
        "PASS: offline generic/PyTorch lock setup, recursive inputs, ABI drift, reuse, credentials, rollback; "
        f"{workbench_environment_result}"
    )


if __name__ == "__main__":
    main()
