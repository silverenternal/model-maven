"""Read-only inventory for optional ML/RL packages and explicit adapters.

Inventory reads distribution metadata in the requested interpreter. Explicit
probe imports only user-selected modules in a subprocess; neither path installs
dependencies, changes the selected interpreter, or chooses a backend.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

from ..protocol import UserError
from ..runtime.entrypoints import configure_project_import_roots
from .common import ENVIRONMENT_CONFIG_PATH, redact_environment_output, run_environment_process, safe_project_path
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
_INVENTORY_SCRIPT = r'''import importlib.metadata as metadata, json, re, sys
normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
requested = set(json.loads(sys.argv[1]))
packages = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    if name:
        key = normalize(name)
        if requested and key not in requested:
            continue
        version = distribution.version
        if not version:
            continue
        # top_level.txt is optional distribution metadata, not proof that an
        # import resolves to this distribution; never import it for inventory.
        try:
            top_level = distribution.read_text("top_level.txt") or ""
            candidates = sorted({line.strip() for line in top_level[:65536].splitlines()
                                 if line.strip().isidentifier() and len(line.strip()) <= 200})[:32]
        except (OSError, UnicodeError, ValueError):
            candidates = []
        packages.setdefault(key, []).append({"name": name, "version": version,
                                             "import_name_candidates": candidates})
print(json.dumps(packages, sort_keys=True))
'''
_PROBE_SCRIPT = r'''import contextlib, importlib, importlib.metadata as metadata, inspect, json, math, os, sys
key = sys.argv[1]
distribution, import_name = json.loads(sys.argv[2])
check_runtime = sys.argv[3] == "1"
project, roots = json.loads(sys.argv[4])
check_devices = sys.argv[5] == "1"
check_layers = sys.argv[6] == "1"
check_operators = sys.argv[7] == "1"
def parameter_details(parameter):
    info = {"name": parameter.name, "kind": parameter.kind.name,
            "required": parameter.default is inspect.Parameter.empty and parameter.kind not in
            (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}
    # Only recognize literal built-in annotations. Never evaluate forward
    # references or call repr() on user-provided annotation objects.
    annotation = parameter.annotation
    known = ((int, "number"), (float, "number"), (str, "text"), (bool, "boolean"),
             (list, "collection"), (tuple, "collection"), (dict, "collection"))
    for builtin, hint in known:
        if annotation is builtin or (type(annotation) is str and annotation == builtin.__name__):
            info["type_hint"] = hint
            break
    value = parameter.default
    # Only bounded primitive defaults are displayed. Do not call repr() on
    # arbitrary objects or copy strings that may contain credentials.
    if value is None or type(value) is bool or (type(value) is int and abs(value) <= 9007199254740991) or (type(value) is float and math.isfinite(value)):
        info["default"] = value
        info["default_available"] = True
    return info
if project in sys.path:
    sys.path.remove(project)
sys.path.insert(0, project)
for entry in reversed(roots):
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
installed_version = None
distribution_status = "not_requested"
if distribution is not None:
    try:
        installed_version = metadata.version(distribution)
        distribution_status = "installed"
    except metadata.PackageNotFoundError:
        distribution_status = "not_installed"
try:
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", installed_version)
            runtime_check_result = None
            device_inventory = None
            layer_catalog = None
            operator_catalog = None
            if check_operators:
                try:
                    candidates = sorted((name, value) for name, value in vars(module).items()
                                        if isinstance(name, str) and name.isidentifier() and not name.startswith("_")
                                        and (inspect.isclass(value) or inspect.isfunction(value) or inspect.isbuiltin(value)))
                    operators = []
                    for name, value in candidates[:2048]:
                        try:
                            params = list(inspect.signature(value).parameters.values())
                            signature = {"status": "observed", "parameters": [parameter_details(p)
                                for p in params[:32]], "truncated": len(params) > 32}
                        except Exception as error:
                            # Avoid exposing exception text, which may contain secrets
                            # from arbitrary user-defined introspection code.
                            signature = {"status": "unavailable", "reason": type(error).__name__[:120]}
                        try:
                            source_module = getattr(value, "__module__", None)
                            if not isinstance(source_module, str):
                                source_module = None
                            elif len(source_module) > 240 or any(ord(c) < 32 for c in source_module):
                                source_module = None
                        except Exception:
                            source_module = None
                        operators.append({"name": name, "reference": import_name + ":" + name,
                                          "kind": "class" if inspect.isclass(value) else "function",
                                          "source_module": source_module,
                                          "signature": signature, "compatibility": "unknown"})
                    operator_catalog = {"status": "observed", "operators": operators,
                                        "truncated": len(candidates) > 2048}
                except Exception as error:
                    operator_catalog = {"status": "failed", "operators": [],
                                        "failure_reason": (type(error).__name__ + ": " + str(error))[:1000]}
            if check_layers and key == "torch":
                try:
                    candidates = sorted((name, value) for name, value in vars(module.nn).items()
                                        if isinstance(name, str) and name.isidentifier()
                                        and not name.startswith("_") and isinstance(value, type)
                                        and issubclass(value, module.nn.Module))
                    layers = []
                    for name, value in candidates[:2048]:
                        try:
                            parameters = list(inspect.signature(value).parameters.values())
                            fields = [parameter_details(p) for p in parameters[:32]]
                            signature = {"status": "observed", "parameters": fields,
                                         "truncated": len(parameters) > 32}
                        except Exception:
                            signature = {"status": "unavailable"}
                        layers.append({"name": name, "signature": signature})
                    layer_catalog = {"status": "observed", "layers": layers,
                                     "truncated": len(candidates) > 2048}
                except Exception as error:
                    layer_catalog = {"status": "failed", "layers": [],
                                     "failure_reason": (type(error).__name__ + ": " + str(error))[:1000]}
            if check_devices and key == "torch":
                devices = [{"id": "cpu", "backend": "cpu", "name": "CPU"}]
                failures = []
                count = 0
                try:
                    cuda = getattr(module, "cuda", None)
                    available = bool(cuda is not None and cuda.is_available())
                    count = int(cuda.device_count()) if available else 0
                    if count < 0:
                        raise ValueError("negative device count")
                    backend = "rocm" if getattr(getattr(module, "version", None), "hip", None) else "cuda"
                    for index in range(min(count, 128)):
                        devices.append({"id": f"cuda:{index}", "backend": backend,
                                        "name": str(cuda.get_device_name(index))[:256]})
                except BaseException as error:
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    failures.append("CUDA: " + type(error).__name__ + ": " + str(error))
                accelerator_status = "unavailable"
                try:
                    accelerator = getattr(module, "accelerator", None)
                    current = getattr(accelerator, "current_accelerator", None)
                    get_count = getattr(accelerator, "device_count", None)
                    if callable(current) and callable(get_count):
                        selected = current(check_available=True)
                        accelerator_status = "observed"
                        if selected is not None and selected.type not in {"cpu", "cuda", "meta"}:
                            accelerator_count = int(get_count())
                            if accelerator_count < 0:
                                raise ValueError("negative accelerator count")
                            if accelerator_count == 0:
                                failures.append("accelerator reported available without visible devices")
                            for index in range(min(accelerator_count, 128)):
                                label = selected.type if accelerator_count == 1 else f"{selected.type}:{index}"
                                devices.append({"id": label, "backend": selected.type, "name": selected.type})
                            if accelerator_count > 128:
                                failures.append("accelerator device list truncated")
                    else:
                        failures.append("generic accelerator query unavailable")
                except BaseException as error:
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    accelerator_status = "failed"
                    failures.append("accelerator: " + type(error).__name__ + ": " + str(error))
                device_inventory = {"status": "partial" if failures or count > 128 else "observed",
                                    "devices": devices, "gpu_count": count, "truncated": count > 128,
                                    "accelerator_query": accelerator_status}
                if failures:
                    device_inventory["failure_reason"] = "; ".join(failures)[:1000]
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
                            raise RuntimeError("JAX arithmetic returned an unexpected value")
                        observed_device = getattr(value, "device", None)
                        if callable(observed_device):
                            observed_device = observed_device()
                        runtime_check_result = {
                            "status": "passed", "profile": "jax-default-array-arithmetic",
                            "device": str(observed_device)[:256] if observed_device is not None else None,
                            "backend": str(jax.default_backend())[:128], "result": scalar,
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
        "distribution_status": distribution_status,
        "distribution_version": installed_version,
        "import_version": str(version)[:200] if version is not None else None,
        "import_origin": (str(origin)[:2000] if (origin := getattr(getattr(module, "__spec__", None), "origin", None)) is not None else None),
    }
    if check_runtime:
        result["runtime_check"] = runtime_check_result
    if check_devices:
        result["device_inventory"] = device_inventory
    if check_layers:
        result["layer_catalog"] = layer_catalog
    if check_operators:
        result["operator_catalog"] = operator_catalog
    print(json.dumps(result))
except KeyboardInterrupt:
    raise
except BaseException as error:
    if isinstance(error, KeyboardInterrupt):
        raise
    print(json.dumps({
        "status": "import_failed",
        "distribution": distribution,
        "distribution_status": distribution_status,
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
    if requested_venv is not None or ENVIRONMENT_CONFIG_PATH.exists():
        raise UserError(
            "ENVIRONMENT_VENV_NOT_FOUND",
            f"指定的项目虚拟环境不可用：{relative}",
            "检查路径，或省略 --venv 使用项目已保存的环境选择。不会静默改查其他解释器。",
        )
    return base_python, "base-python"


def _requested_libraries(names):
    if names is None or names == []:
        return []
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
        if not isinstance(name, str) or not name or len(name) > 200:
            raise UserError("ENVIRONMENT_LIBRARY_QUERY_INVALID", f"无效 Python 库或模块名称：{name!r}。")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if all(part.isidentifier() for part in name.split(".")):
            # Valid Python module names are case-sensitive. Map a known
            # distribution only when its documented import name matches exactly.
            key = next((candidate for candidate, definition in _LIBRARIES.items()
                        if definition["import_name"] == name), name)
        elif normalized in _LIBRARIES:
            key = normalized
        else:
            raise UserError("ENVIRONMENT_LIBRARY_QUERY_INVALID", f"未知库请使用 Python 模块名：{name!r}。")
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def inspect_environment_libraries(request):
    """Report installed package metadata without importing optional packages."""
    runtime = request.get("runtime", {})
    base_python = runtime.get("python_executable") or sys.executable
    if not base_python:
        raise UserError("PYTHON_SELECTION_REQUIRED", "无法确定当前 Python 解释器，请明确选择。")
    target_python, selected_by = _selected_python(
        base_python,
        request.get("python_explicit") is True,
        request.get("venv"),
    )
    requested = request.get("libraries")
    explicit = requested is not None and requested != []
    requested_names = _requested_libraries(requested) if explicit else []
    result = run_environment_process(
        [target_python, "-c", _INVENTORY_SCRIPT, json.dumps(requested_names)],
        "已安装 ML/RL 库元数据读取",
        timeout=60,
    )
    try:
        inventory = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise UserError("ENVIRONMENT_LIBRARY_INVENTORY_INVALID", "Python 没有返回有效的 distribution 元数据。") from error
    if not isinstance(inventory, dict) or len(inventory) > 100_000:
        raise UserError("ENVIRONMENT_LIBRARY_INVENTORY_INVALID", "Python distribution 元数据超出可接受范围。")

    if not explicit:
        # List only distributions observed in the selected interpreter. Known
        # adapter descriptions enrich matching entries but never add packages.
        discovered = sorted(inventory)
        selected = discovered[:4096]
        truncated = max(0, len(discovered) - len(selected))
    else:
        selected = requested_names
        truncated = 0
    libraries = []
    for key in selected:
        definition = _LIBRARIES.get(key, {
            "distribution": key,
            "import_name": None,
            "backend": None,
            "adapter_status": "unknown",
            "summary": "Discovered or user-declared distribution; no first-party adapter claim is made.",
        })
        matches = inventory.get(key, [])
        if not isinstance(matches, list):
            matches = []
        candidates = sorted({candidate for match in matches if isinstance(match, dict)
                             for candidate in match.get("import_name_candidates", [])
                             if isinstance(candidate, str) and candidate.isidentifier()})[:32]
        hint = definition["import_name"]
        hint_source = "built-in-adapter" if hint else None
        if not hint and len(candidates) == 1:
            hint, hint_source = candidates[0], "distribution-metadata-unverified"
        libraries.append({
            "name": key,
            "distribution": definition["distribution"],
            "import_name_hint": hint,
            "import_name_hint_source": hint_source,
            "import_name_candidates": candidates,
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
        "omitted_count": truncated,
        "discovery": "selected-interpreter-distribution-metadata",
        "note": "发现只读取 package metadata，不执行库代码；已安装不代表存在 Model-Maven 专用适配器。依赖安装和后端选择都必须由用户显式完成。",
    }


def probe_environment_libraries(request):
    """Import only explicitly named libraries or modules in the selected Python."""
    runtime = request.get("runtime", {})
    base_python = runtime.get("python_executable") or sys.executable
    if not base_python:
        raise UserError("PYTHON_SELECTION_REQUIRED", "无法确定当前 Python 解释器，请明确选择。")
    target_python, selected_by = _selected_python(
        base_python,
        request.get("python_explicit") is True,
        request.get("venv"),
    )
    check_runtime = request.get("check_runtime") is True
    check_devices = request.get("device_inventory") is True
    check_layers = request.get("layer_catalog") is True
    check_operators = request.get("operator_catalog") is True
    selected_roots = configure_project_import_roots(request)
    project_root = Path(request.get("project_root") or Path.cwd()).resolve(strict=True)
    requested_cwd = runtime.get("working_directory")
    if requested_cwd is not None:
        if not isinstance(requested_cwd, str) or not Path(requested_cwd).is_absolute():
            raise UserError("WORKING_DIRECTORY_INVALID", "运行工作目录必须是绝对路径。")
        try:
            cwd = Path(requested_cwd).resolve(strict=True)
            if cwd != Path(requested_cwd) or not cwd.is_dir():
                raise ValueError("working directory changed or is not a directory")
        except (OSError, ValueError, RuntimeError) as error:
            raise UserError("WORKING_DIRECTORY_UNAVAILABLE", "运行工作目录不可用或已变化。") from error
    else:
        cwd = Path.cwd()
    # Resolve relative executable choices in the original project context before
    # changing the subprocess cwd. Do not resolve symlinks: venvs use them.
    if requested_cwd is not None:
        located_python = shutil.which(target_python)
        if located_python:
            target_python = os.path.abspath(located_python)
        elif os.path.dirname(target_python):
            target_python = os.path.abspath(target_python)
    requested = _requested_probe_libraries(request.get("libraries"))
    if check_layers and requested != ["torch"]:
        raise UserError("ENVIRONMENT_LAYER_CATALOG_INVALID", "Layer 清单要求单独显式指定 --library torch。")
    if check_runtime and len(requested) > 4:
        raise UserError(
            "ENVIRONMENT_RUNTIME_CHECK_LIMIT",
            "单次 --check-runtime 最多检查 4 个明确点名的库。",
            "分批执行；每个库都在所选解释器的独立子进程中初始化。",
        )
    probes = []
    for key in requested:
        definition = _LIBRARIES.get(key, {
            "distribution": None, "import_name": key, "backend": None,
            "adapter_status": "unknown",
        })
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
                json.dumps([str(project_root), selected_roots], ensure_ascii=True),
                "1" if check_devices else "0",
                "1" if check_layers else "0",
                "1" if check_operators else "0",
            ],
            f"显式 Python 库导入探测：{key}",
            timeout=120,
            working_directory=cwd,
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
        if value.get("distribution_status") in {"not_requested", "installed", "not_installed"}:
            probe["distribution_status"] = value["distribution_status"]
        if isinstance(value.get("import_version"), str):
            probe["import_version"] = value["import_version"][:200]
        if isinstance(value.get("import_origin"), str):
            probe["import_origin"] = value["import_origin"][:2000]
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
        inventory = value.get("device_inventory")
        if check_devices:
            if isinstance(inventory, dict):
                inventory = dict(inventory)
                if isinstance(inventory.get("failure_reason"), str):
                    inventory["failure_reason"] = redact_environment_output(inventory["failure_reason"], limit=1000)
                probe["device_inventory"] = inventory
            else:
                probe["device_inventory"] = {"status": "unknown", "devices": [], "reason": "import_unavailable"}
        if check_layers:
            catalog = value.get("layer_catalog")
            if isinstance(catalog, dict):
                catalog = dict(catalog)
                if isinstance(catalog.get("failure_reason"), str):
                    catalog["failure_reason"] = redact_environment_output(catalog["failure_reason"], limit=1000)
                probe["layer_catalog"] = catalog
            else:
                probe["layer_catalog"] = {"status": "unknown", "layers": [], "reason": "import_unavailable"}
        if check_operators:
            catalog = value.get("operator_catalog")
            if isinstance(catalog, dict):
                catalog = dict(catalog)
                if isinstance(catalog.get("failure_reason"), str):
                    catalog["failure_reason"] = redact_environment_output(catalog["failure_reason"], limit=1000)
                probe["operator_catalog"] = catalog
            else:
                probe["operator_catalog"] = {"status": "unknown", "operators": [], "reason": "import_unavailable"}
        probes.append(probe)
    return {
        "python_executable": target_python,
        "selected_by": selected_by,
        "working_directory": str(cwd),
        "project_import_roots": runtime.get("project_import_roots"),
        "inspection": "explicit-device-query" if check_devices else ("explicit-import-and-runtime-smoke" if check_runtime else "explicit-import"),
        # A successful import may resolve to project code, and a failed import
        # may have already executed module initializers. Neither result proves
        # whether third-party code ran.
        "third_party_code_imported": None,
        "third_party_import_status": "unknown",
        "import_attempted": bool(probes),
        "import_may_have_executed_code": bool(probes),
        "auto_installed": False,
        "auto_bound": False,
        "probes": probes,
        "note": ("已显式导入 PyTorch 并查询当前设备可见性；可能初始化驱动，设备之后可能变化。未安装依赖、未选择设备。"
                 if check_devices else
                 "仅导入用户通过 --library 点名的库或 Python 模块；发行包元数据和实际导入来源分别报告，前者不证明后者归属。--check-runtime 只对支持的库运行有限计算，JAX 使用环境默认后端；其它模块返回 unsupported。导入成功不代表提供专用模型或训练后端。未安装依赖、未切换解释器、未绑定 Trainer。"),
    }
