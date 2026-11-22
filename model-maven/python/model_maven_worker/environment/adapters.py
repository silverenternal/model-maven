"""Read-only inventory for optional ML/RL packages and explicit adapters.

This module inspects distribution metadata in the requested interpreter. It
never imports the listed packages, installs dependencies, changes the selected
interpreter, or chooses an execution backend.
"""

import json
import re
from pathlib import Path

from ..protocol import UserError
from .common import redact_environment_output, run_environment_process, safe_project_path
from .setup import selected_environment_venv, virtualenv_python


_LIBRARIES = {
    "torch": {
        "distribution": "torch",
        "import_name": "torch",
        "backend": "pytorch-native",
        "adapter_status": "implemented",
        "summary": "Native Model-Maven model inspection, probes, and supported training modes.",
    },
    "torchvision": {
        "distribution": "torchvision",
        "import_name": "torchvision",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "May be imported by explicitly selected user model/data code.",
    },
    "timm": {
        "distribution": "timm",
        "import_name": "timm",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "May be imported by explicitly selected user model code.",
    },
    "lightning": {
        "distribution": "lightning",
        "import_name": "lightning",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "User code can import it; Model-Maven does not auto-bind Lightning Trainer.",
    },
    "pytorch-lightning": {
        "distribution": "pytorch-lightning",
        "import_name": "pytorch_lightning",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "User code can import it; Model-Maven does not auto-bind Lightning Trainer.",
    },
    "transformers": {
        "distribution": "transformers",
        "import_name": "transformers",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "May be imported by explicitly selected user code; no HF Trainer binding is inferred.",
    },
    "diffusers": {
        "distribution": "diffusers",
        "import_name": "diffusers",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "May be imported by explicitly selected user code.",
    },
    "jax": {
        "distribution": "jax",
        "import_name": "jax",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "Package presence does not imply JAX model execution support.",
    },
    "flax": {
        "distribution": "flax",
        "import_name": "flax",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "Package presence does not imply Flax model execution support.",
    },
    "gymnasium": {
        "distribution": "gymnasium",
        "import_name": "gymnasium",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "Environment package can be used by user code; RL loop is not auto-selected.",
    },
    "stable-baselines3": {
        "distribution": "stable-baselines3",
        "import_name": "stable_baselines3",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "Package presence does not imply Stable-Baselines3 training integration.",
    },
    "ray": {
        "distribution": "ray",
        "import_name": "ray",
        "backend": None,
        "adapter_status": "package-only",
        "summary": "Package presence does not imply RLlib execution support.",
    },
}

_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
_INVENTORY_SCRIPT = r'''import importlib.metadata as metadata, json, re
normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
packages = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    version = distribution.version
    if name and version:
        key = normalize(name)
        packages.setdefault(key, []).append({"name": name, "version": version})
print(json.dumps(packages, sort_keys=True))
'''
_PROBE_SCRIPT = r'''import contextlib, importlib, importlib.metadata as metadata, json, math, os, sys
key = sys.argv[1]
distribution, import_name = json.loads(sys.argv[2])
check_runtime = sys.argv[3] == "1"
try:
    installed_version = metadata.version(distribution)
except metadata.PackageNotFoundError:
    result = {"status": "not_installed", "distribution": distribution}
    if check_runtime:
        result["runtime_check"] = {"status": "not_run", "reason": "package_not_installed"}
    print(json.dumps(result))
    raise SystemExit(0)
try:
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            if check_runtime and key == "jax":
                os.environ["JAX_PLATFORMS"] = "cpu"
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", installed_version)
            runtime_check_result = None
            if check_runtime:
                try:
                    if key == "torch":
                        import torch
                        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cpu")
                        value = (tensor * tensor).sum()
                        scalar = float(value.item())
                        if not math.isfinite(scalar) or scalar != 5.0:
                            raise RuntimeError("CPU tensor arithmetic returned an unexpected value")
                        runtime_check_result = {
                            "status": "passed", "profile": "torch-cpu-tensor-arithmetic",
                            "device": "cpu", "result": scalar,
                        }
                    elif key == "jax":
                        jax = importlib.import_module("jax")
                        jnp = importlib.import_module("jax.numpy")
                        array = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
                        value = jnp.sum(array * array)
                        value.block_until_ready()
                        scalar = float(value)
                        if not math.isfinite(scalar) or scalar != 5.0:
                            raise RuntimeError("CPU JAX arithmetic returned an unexpected value")
                        runtime_check_result = {
                            "status": "passed", "profile": "jax-cpu-array-arithmetic",
                            "device": "cpu", "result": scalar,
                            "runtime_version": str(getattr(jax, "__version__", installed_version))[:200],
                        }
                    elif key == "torchvision":
                        import torch
                        import torchvision
                        boxes = torch.tensor(
                            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.9, 0.9]],
                            dtype=torch.float32, device="cpu",
                        )
                        scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device="cpu")
                        kept = torchvision.ops.nms(boxes, scores, 0.5).tolist()
                        if kept != [0]:
                            raise RuntimeError("CPU torchvision NMS returned an unexpected result")
                        runtime_check_result = {
                            "status": "passed", "profile": "torchvision-cpu-nms",
                            "device": "cpu", "kept_indices": kept,
                            "runtime_version": str(getattr(torchvision, "__version__", installed_version))[:200],
                        }
                    elif key == "gymnasium":
                        import numpy as np
                        gymnasium = importlib.import_module("gymnasium")
                        sample = gymnasium.spaces.Box(
                            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                        ).sample()
                        if sample.shape != (2,) or not bool(np.isfinite(sample).all()):
                            raise RuntimeError("Gymnasium Box sample did not satisfy its declared bounds")
                        runtime_check_result = {
                            "status": "passed", "profile": "gymnasium-box-sample",
                            "shape": list(sample.shape),
                        }
                    else:
                        runtime_check_result = {
                            "status": "unsupported",
                            "reason": "No first-party, side-effect-bounded runtime smoke is defined for this library.",
                        }
                except KeyboardInterrupt:
                    raise
                except BaseException as error:
                    runtime_check_result = {
                        "status": "failed",
                        "failure_type": type(error).__name__[:120],
                        "failure_reason": str(error)[:2000],
                    }
    result = {
        "status": "imported",
        "distribution": distribution,
        "distribution_version": installed_version,
        "import_version": str(version)[:200],
    }
    if check_runtime:
        result["runtime_check"] = runtime_check_result
    print(json.dumps(result))
except KeyboardInterrupt:
    raise
except BaseException as error:
    if isinstance(error, KeyboardInterrupt):
        raise
    print(json.dumps({
        "status": "import_failed",
        "distribution": distribution,
        "distribution_version": installed_version,
        "failure_type": type(error).__name__[:120],
        "failure_reason": str(error)[:2000],
    }))
'''


def _selected_python(base_python, explicit_python, requested_venv=None):
    if explicit_python:
        return base_python, "explicit-python"
    relative = selected_environment_venv(requested_venv)
    venv = safe_project_path(relative, "虚拟环境")
    candidate = virtualenv_python(venv)
    if venv.is_dir() and candidate.is_file():
        return candidate.as_posix(), relative
    if requested_venv is not None:
        raise UserError(
            "ENVIRONMENT_VENV_NOT_FOUND",
            f"指定的项目虚拟环境不可用：{relative}",
            "检查路径，或省略 --venv 使用项目已保存的环境选择。不会静默改查其他解释器。",
        )
    return base_python, "base-python"


def _requested_libraries(names):
    if names is None or names == []:
        return list(_LIBRARIES)
    if not isinstance(names, list) or len(names) > 128:
        raise UserError("ENVIRONMENT_LIBRARY_QUERY_INVALID", "library 查询必须是最多 128 项的名称数组。")
    result = []
    seen = set()
    for name in names:
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise UserError("ENVIRONMENT_LIBRARY_QUERY_INVALID", f"无效 Python distribution 名称：{name!r}。")
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _requested_probe_libraries(names):
    if not isinstance(names, list) or not names or len(names) > 32:
        raise UserError(
            "ENVIRONMENT_LIBRARY_PROBE_REQUIRED",
            "probe 必须通过一个或多个 --library 明确指定要导入的候选库。",
            "例如：m-model env probe --library gymnasium。不会默认导入任何库。",
        )
    result = []
    seen = set()
    for name in names:
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise UserError("ENVIRONMENT_LIBRARY_QUERY_INVALID", f"无效 Python distribution 名称：{name!r}。")
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key not in _LIBRARIES:
            raise UserError(
                "ENVIRONMENT_LIBRARY_PROBE_UNSUPPORTED",
                f"库 {name!r} 不在可显式 probe 的安全候选列表中。",
                f"可选候选：{', '.join(sorted(_LIBRARIES))}。任意包仍可用 env libraries 查询 metadata。",
            )
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def inspect_environment_libraries(request):
    """Report installed package metadata without importing optional packages."""
    runtime = request.get("runtime", {})
    base_python = runtime.get("python_executable") or "python3"
    target_python, selected_by = _selected_python(
        base_python,
        request.get("python_explicit") is True,
        request.get("venv"),
    )
    result = run_environment_process(
        [target_python, "-c", _INVENTORY_SCRIPT],
        "已安装 ML/RL 库元数据读取",
        timeout=60,
    )
    try:
        inventory = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise UserError("ENVIRONMENT_LIBRARY_INVENTORY_INVALID", "Python 没有返回有效的 distribution 元数据。") from error
    if not isinstance(inventory, dict) or len(inventory) > 100_000:
        raise UserError("ENVIRONMENT_LIBRARY_INVENTORY_INVALID", "Python distribution 元数据超出可接受范围。")

    libraries = []
    for key in _requested_libraries(request.get("libraries")):
        definition = _LIBRARIES.get(key, {
            "distribution": key,
            "import_name": None,
            "backend": None,
            "adapter_status": "unknown",
            "summary": "User-declared distribution; no first-party adapter claim is made.",
        })
        matches = inventory.get(key, [])
        if not isinstance(matches, list):
            matches = []
        libraries.append({
            "name": key,
            "distribution": definition["distribution"],
            "import_name_hint": definition["import_name"],
            "installed": bool(matches),
            "installed_versions": matches,
            "adapter_status": definition["adapter_status"],
            "backend": definition["backend"],
            "auto_installed": False,
            "auto_bound": False,
            "summary": definition["summary"],
        })
    return {
        "python_executable": target_python,
        "selected_by": selected_by,
        "inspection": "distribution-metadata-only",
        "third_party_code_imported": False,
        "libraries": libraries,
        "note": "发现只读取 package metadata，不执行库代码；已安装不代表存在 Model-Maven 专用适配器。依赖安装和后端选择都必须由用户显式完成。",
    }


def probe_environment_libraries(request):
    """Import only explicitly named, allowlisted candidates in the selected Python."""
    runtime = request.get("runtime", {})
    base_python = runtime.get("python_executable") or "python3"
    target_python, selected_by = _selected_python(
        base_python,
        request.get("python_explicit") is True,
        request.get("venv"),
    )
    check_runtime = request.get("check_runtime") is True
    requested = _requested_probe_libraries(request.get("libraries"))
    if check_runtime and len(requested) > 4:
        raise UserError(
            "ENVIRONMENT_RUNTIME_CHECK_LIMIT",
            "单次 --check-runtime 最多检查 4 个明确点名的库。",
            "分批执行；每个库都在所选解释器的独立子进程中初始化。",
        )
    probes = []
    for key in requested:
        definition = _LIBRARIES[key]
        result = run_environment_process(
            [
                target_python,
                "-c",
                _PROBE_SCRIPT,
                key,
                json.dumps(
                    [definition["distribution"], definition["import_name"]],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                "1" if check_runtime else "0",
            ],
            f"显式 Python 库导入探测：{key}",
            timeout=120,
        )
        try:
            value = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise UserError(
                "ENVIRONMENT_LIBRARY_PROBE_INVALID",
                f"库 {key} 的 probe 子进程没有返回有效结果。",
            ) from error
        if not isinstance(value, dict) or value.get("status") not in {
            "not_installed", "imported", "import_failed"
        }:
            raise UserError("ENVIRONMENT_LIBRARY_PROBE_INVALID", f"库 {key} 的 probe 结果格式无效。")
        probe = {
            "library": key,
            "distribution": definition["distribution"],
            "import_name": definition["import_name"],
            "status": value["status"],
            "adapter_status": definition["adapter_status"],
            "backend": definition["backend"],
            "auto_installed": False,
            "auto_bound": False,
        }
        if isinstance(value.get("distribution_version"), str):
            probe["distribution_version"] = value["distribution_version"][:200]
        if isinstance(value.get("import_version"), str):
            probe["import_version"] = value["import_version"][:200]
        if isinstance(value.get("failure_type"), str):
            probe["failure_type"] = value["failure_type"][:120]
        if isinstance(value.get("failure_reason"), str):
            probe["failure_reason"] = redact_environment_output(value["failure_reason"], limit=1000)
        runtime_result = value.get("runtime_check")
        if isinstance(runtime_result, dict):
            runtime_check = {key: item for key, item in runtime_result.items() if key != "failure_reason"}
            if isinstance(runtime_result.get("failure_reason"), str):
                runtime_check["failure_reason"] = redact_environment_output(
                    runtime_result["failure_reason"], limit=1000
                )
            probe["runtime_check"] = runtime_check
        elif check_runtime:
            probe["runtime_check"] = {
                "status": "not_run",
                "reason": "import_failed" if value["status"] == "import_failed" else "runtime_result_unavailable",
            }
        probes.append(probe)
    return {
        "python_executable": target_python,
        "selected_by": selected_by,
        "inspection": "explicit-allowlisted-import-and-runtime-smoke" if check_runtime else "explicit-allowlisted-import",
        "third_party_code_imported": any(item["status"] == "imported" for item in probes),
        "auto_installed": False,
        "auto_bound": False,
        "probes": probes,
        "note": "仅导入用户通过 --library 点名且属于候选清单的包。--check-runtime 只对有固定 CPU smoke 的库运行有限运算；其它库返回 unsupported，不代表失败。通过 smoke 不代表 Model-Maven 提供该库的模型或训练后端。未安装依赖、未切换解释器、未绑定 Trainer。",
    }
