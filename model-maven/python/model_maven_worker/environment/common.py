"""Shared path, process, and environment identity helpers."""

import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ..protocol import UserError


ENVIRONMENT_LOCK_SCHEMA = 1
ENVIRONMENT_LOCK_PATH = Path("requirements.lock.txt")
ENVIRONMENT_METADATA_PATH = Path(".model-maven/environment.lock.json")
ENVIRONMENT_CONFIG_PATH = Path(".model-maven/environment.json")
PYTHON_IDENTITY_SCRIPT = (
    "import json,platform,sys,sysconfig; "
    "print(json.dumps({'implementation':sys.implementation.name,'version':sys.version.split()[0],"
    "'cache_tag':sys.implementation.cache_tag,'platform':sysconfig.get_platform(),"
    "'machine':platform.machine(),'soabi':sysconfig.get_config_var('SOABI')}))"
)

def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def distribution_name_key(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def environment_fingerprint(
    python,
    pytorch_version,
    pytorch_cuda_version,
    distributions,
    ambiguous,
    issues,
):
    return digest({
        "schema_version": 1,
        "python": python,
        "pytorch_version": pytorch_version,
        "pytorch_cuda_version": pytorch_cuda_version,
        "distributions": distributions,
        "ambiguous_distribution_names": ambiguous,
        "issues": issues,
    })


def safe_project_path(value, label, must_exist=False):
    if not isinstance(value, str) or not value:
        raise UserError("ENVIRONMENT_PATH_INVALID", f"{label} 必须是项目内相对路径。")
    if os.name == "nt":
        value = value.replace("\\", "/")
    elif "\\" in value:
        raise UserError("ENVIRONMENT_PATH_INVALID", f"{label} 必须使用相对 POSIX 路径。")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise UserError("ENVIRONMENT_PATH_INVALID", f"{label} 不能使用绝对路径或目录跳转。")
    root = Path.cwd().resolve(strict=True)
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise UserError("ENVIRONMENT_PATH_UNSAFE", f"{label} 路径包含符号链接：{relative}")
    try:
        candidate.resolve(strict=must_exist).relative_to(root)
    except FileNotFoundError as error:
        if must_exist:
            raise UserError(
                "ENVIRONMENT_INPUT_MISSING",
                f"找不到{label}：{relative}",
                "在项目中创建 requirements.txt 并列出需要的运行依赖（包括 PyTorch），然后重试。",
            ) from error
        raise UserError("ENVIRONMENT_PATH_UNSAFE", f"{label} 路径无效：{relative}") from error
    except (OSError, ValueError) as error:
        raise UserError("ENVIRONMENT_PATH_UNSAFE", f"{label} 路径不在当前项目内：{relative}") from error
    if must_exist and not candidate.is_file():
        raise UserError("ENVIRONMENT_INPUT_INVALID", f"{label} 不是普通文件：{relative}")
    return candidate


def safe_output_path(relative, label):
    candidate = safe_project_path(relative, label)
    parent = candidate.parent
    cursor = Path.cwd().resolve(strict=True)
    for part in parent.relative_to(cursor).parts:
        cursor = cursor / part
        if cursor.exists() and (cursor.is_symlink() or not cursor.is_dir()):
            raise UserError("ENVIRONMENT_PATH_UNSAFE", f"{label} 的父路径不安全：{relative}")
    if candidate.exists() and (candidate.is_symlink() or not candidate.is_file()):
        raise UserError("ENVIRONMENT_PATH_UNSAFE", f"{label} 不是普通文件：{relative}")
    return candidate


def atomic_write_bytes(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def redact_environment_output(value, limit=4000):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = re.sub(r"(?i)\b(?:https?|file)://[^\s\"'<>]+", "<URL redacted>", str(value))
    value = re.sub(
        r"(?i)([?&](?:token|password|passwd|secret|signature|credential|access_key)=)[^&\s]+",
        r"\1<redacted>",
        value,
    )
    value = re.sub(r"(?i)\b(token|password|passwd|secret|api[_-]?key)\s*[:=]\s*[^\s,;]+", r"\1=<redacted>", value)
    value = value.strip()
    if len(value) > limit:
        value = value[:limit] + "… [输出已截断]"
    return value


def run_environment_process(arguments, label, timeout=1800, stream_output=False, source_build=False):
    environment = os.environ.copy()
    environment.update({
        "PIP_NO_INPUT": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    try:
        if stream_output:
            process = subprocess.Popen(
                arguments,
                cwd=Path.cwd(),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            lines = queue.Queue()

            def read_output():
                try:
                    for line in process.stdout:
                        lines.put(line)
                finally:
                    lines.put(None)

            threading.Thread(target=read_output, daemon=True).start()
            captured = []
            capture_size = 0
            reader_done = False
            deadline = time.monotonic() + timeout
            while not reader_done or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 and process.poll() is None:
                    process.kill()
                    process.wait()
                    detail = redact_environment_output("".join(captured)[-4000:])
                    raise UserError("ENVIRONMENT_TIMEOUT", f"{label} 超时。" + (f" {detail}" if detail else ""), "检查网络和 pip 缓存后重试；可用 --timeout 调整时限。")
                try:
                    line = lines.get(timeout=max(0.05, min(0.2, remaining)))
                except queue.Empty:
                    continue
                if line is None:
                    reader_done = True
                    continue
                if capture_size < 1024 * 1024:
                    captured.append(line)
                    capture_size += len(line.encode("utf-8", errors="replace"))
                safe_line = redact_environment_output(line, limit=8192)
                if safe_line:
                    sys.stderr.write(safe_line)
                    if not safe_line.endswith(("\n", "\r")):
                        sys.stderr.write("\n")
                    sys.stderr.flush()
            result = subprocess.CompletedProcess(arguments, process.wait(), stdout="".join(captured), stderr="")
        else:
            result = subprocess.run(
                arguments,
                cwd=Path.cwd(),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
    except FileNotFoundError as error:
        raise UserError("PYTHON_UNAVAILABLE", f"找不到环境工具：{arguments[0]}", "用 --python 指定可用的 Python 解释器。") from error
    except subprocess.TimeoutExpired as error:
        detail = redact_environment_output(error.stderr)
        raise UserError("ENVIRONMENT_TIMEOUT", f"{label} 超时。" + (f" {detail}" if detail else ""), "检查网络和 pip 缓存后重试；可用 --timeout 调整时限。") from error
    except OSError as error:
        raise UserError("ENVIRONMENT_PROCESS_FAILED", f"无法运行{label}：{error}") from error
    if result.returncode != 0:
        detail = redact_environment_output(result.stderr or result.stdout)
        raise UserError(
            "ENVIRONMENT_PROCESS_FAILED",
            f"{label}失败（退出码 {result.returncode}）。" + (f" {detail}" if detail else ""),
            (
                "检查 Python/pip、构建后端与包索引配置；该命令执行了项目中声明的 PEP 517 构建后端。"
                if source_build
                else "检查 Python/pip、依赖约束和包索引配置；环境锁解析不会执行源码构建。"
            ),
        )
    return result


def python_identity(executable):
    result = run_environment_process([executable, "-c", PYTHON_IDENTITY_SCRIPT], "Python 信息读取", timeout=30)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise UserError("PYTHON_IDENTITY_INVALID", "Python 没有返回有效的运行时信息。") from error
    required = {"implementation", "version", "cache_tag", "platform", "machine", "soabi"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise UserError("PYTHON_IDENTITY_INVALID", "Python 运行时信息缺少必要字段。")
    return value

