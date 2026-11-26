"""Reconstruct a model from project inputs and replay a validated edit variant."""

import inspect
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError
from ..edit.initialization_rng import layer_rng_scope, validate_rng_devices
from ..runtime.entrypoints import configure_project_import_roots
from .structure import module_entries, resolve_module


def _declared_route_evidence(model):
    """Read optional model-owned declarations; they never establish runtime flow."""
    declared = vars(model).get("_model_maven_declared_routes")
    if declared is None:
        return None
    result = {"scope": "model_self_reported_at_construction", "status": "unavailable", "routes": []}
    if type(declared) is not list or len(declared) > 32:
        return result
    try:
        _, aliases = module_entries(model)
        for entry in declared:
            if type(entry) is not dict:
                result["routes"] = []
                return result
            slot, source, path = (entry.get(key) for key in ("slot", "input", "module_path"))
            if any(type(value) is not str or not value or len(value) > 128 for value in (slot, source, path)):
                result["routes"] = []
                return result
            target = resolve_module(model, path)
            result["routes"].append({"slot": slot, "input": source, "registered_paths": aliases[id(target)]})
    except Exception:
        result["routes"] = []
        return result
    result["status"] = "available"
    return result


@dataclass(frozen=True)
class ModelLoadingRuntime:
    torch: Any
    validate_variant: Callable[[dict], Any]
    validate_model_injections: Callable[[Any], dict]
    verify_manifest: Callable[[dict], Any]
    verify_model_injection_references: Callable[[dict, str], Any]
    entrypoint: Callable[[str], Callable[..., Any]]
    construct_model: Callable[..., Any]
    verify_component_references: Callable[..., Any]
    apply_history_item: Callable[..., Any]


def load_model(runtime, request, *, reset_rng=True):
    observation_operations = {"inspect", "layers", "probe", "verify", "semantic_mark", "semantic_supervise"}
    variant = request.get("variant")
    initialization_policy = request.get("model_initialization_policy")
    initialization_seed = request.get("model_initialization_seed")
    if initialization_policy is not None and (
        not isinstance(initialization_policy, str)
        or initialization_policy not in {"runtime", "seeded"}
    ):
        raise UserError("MODEL_INITIALIZATION_POLICY_INVALID", "模型构造随机策略必须是 runtime 或 seeded。")
    if initialization_policy is None and initialization_seed is not None:
        raise UserError("MODEL_INITIALIZATION_POLICY_REQUIRED", "单独指定构造种子时必须同时指定 model_initialization_policy=seeded。")
    if initialization_policy == "runtime" and initialization_seed is not None:
        raise UserError("MODEL_INITIALIZATION_POLICY_CONFLICT", "runtime 构造策略不能同时指定 model_initialization_seed。")
    if initialization_seed is not None and (
        type(initialization_seed) is not int or not 0 <= initialization_seed < 2**64
    ):
        raise UserError("MODEL_INITIALIZATION_SEED_INVALID", "模型构造种子必须是无符号 64 位整数。")
    if not reset_rng and (
        initialization_policy == "seeded"
        or (initialization_policy == "runtime" and request.get("seed") is not None)
    ):
        raise UserError("MODEL_INITIALIZATION_RNG_RESET_DISABLED", "当前重建路径禁止重置随机状态，无法执行请求的构造/执行种子策略。",
                        "在允许 RNG 重置的独立运行中执行，或省略显式策略和执行种子。")
    requested_policy = request.get("variant_device_policy")
    base_rng_devices = request.get("variant_base_rng_devices")
    if base_rng_devices is not None:
        validate_rng_devices(base_rng_devices)
        if not variant and request.get("operation") != "edit":
            raise UserError("VARIANT_BASE_RNG_UNSUPPORTED", "基础随机设备只适用于新变体编辑。")
    if requested_policy is not None and (not isinstance(requested_policy, str) or requested_policy not in {"legacy_cpu", "factory"}):
        raise UserError("VARIANT_DEVICE_POLICY_INVALID", "不支持的变体设备策略。")
    if variant:
        if initialization_policy is not None or initialization_seed is not None:
            raise UserError("MODEL_INITIALIZATION_POLICY_CONFLICT", "已有变体使用已记录的构造种子重建，不能覆盖其初始化策略。",
                            "沿用变体种子，或建立新变体。")
        runtime.validate_variant(variant)
        saved_base_rng_devices = variant.get("base_rng_devices")
        if base_rng_devices is not None and base_rng_devices != saved_base_rng_devices:
            raise UserError("VARIANT_BASE_RNG_CONFLICT", "不能更改已有变体的基础模型随机设备。", "沿用已记录策略或创建新变体。")
        base_rng_devices = saved_base_rng_devices
        device_policy = variant.get("device_policy", "legacy_cpu")
        if requested_policy is not None and requested_policy != device_policy:
            raise UserError("VARIANT_DEVICE_POLICY_CONFLICT", "不能更改已有变体的设备重建策略。", "请选择新变体。")
        origin = variant["origin"]
        if request.get("model") and (
            request["model"] != origin["model"]
            or request.get("model_params", {}) != origin["params"]
        ):
            raise UserError(
                "VARIANT_ORIGIN_CONFLICT",
                "显式模型参数与已有变体来源不一致。",
                "使用新的 --variant 名称。",
            )
        origin_injections = runtime.validate_model_injections(
            origin.get("model_injections", {})
        )
        requested_injections = runtime.validate_model_injections(
            request.get("model_injections", {})
        )
        if requested_injections and requested_injections != origin_injections:
            raise UserError(
                "VARIANT_ORIGIN_CONFLICT",
                "显式组件接缝与已有变体固定的模型装配不一致。",
                "沿用变体原始接缝，或选择新变体名称重新构造。",
            )
        if (
            request.get("operation") == "edit"
            and request.get("seed") is not None
            and request["seed"] != variant["seed"]
        ):
            raise UserError(
                "VARIANT_SEED_CONFLICT",
                f"当前变体使用 seed={variant['seed']} 构造模型和编辑层；本次 seed={request['seed']}。",
                "保留当前变体时省略 --seed。要用新种子初始化编辑，请选择新变体名，例如 --variant experiment --seed 1234；train/try 的 --seed 只控制本次运行随机性。",
            )
        if variant["torch_version"] != runtime.torch.__version__:
            raise UserError(
                "ENVIRONMENT_MISMATCH",
                "PyTorch 版本与变体创建环境不同。",
                "请用原版本或新名称建立变体。",
            )
        root = origin["project_root"]
        runtime.verify_manifest(variant["source_manifest"])
        seed = variant["seed"]
        construction_seed = seed
        effective_initialization_policy = "seeded"
        model_ref, params = origin["model"], origin["params"]
        injections = origin_injections
    else:
        device_policy = requested_policy or "legacy_cpu"
        root = request["project_root"]
        # Preserve the legacy default until a caller explicitly selects a policy.
        seed = request.get("seed")
        source_edit = (request.get("operation") == "edit"
                       and isinstance(request.get("edit"), dict)
                       and request["edit"].get("target") == "source")
        if (seed is None and initialization_policy is None
                and request.get("operation") not in observation_operations and not source_edit):
            seed = 42
        if request.get("operation") == "edit" and not source_edit:
            if initialization_policy == "runtime":
                raise UserError("MODEL_INITIALIZATION_POLICY_CONFLICT", "新变体必须固定可重建的模型构造种子，不能使用 runtime 随机状态。",
                                "选择 seeded 并让变体记录同一构造种子。")
            if initialization_seed is not None and initialization_seed != seed:
                raise UserError("MODEL_INITIALIZATION_SEED_CONFLICT", "新变体的模型构造种子必须与变体种子相同。",
                                "变体种子同时标识基础模型和编辑历史；请统一两个种子。")
        if initialization_policy == "seeded":
            construction_seed = initialization_seed if initialization_seed is not None else seed
            if construction_seed is None:
                raise UserError("MODEL_INITIALIZATION_SEED_REQUIRED", "seeded 构造策略需要 model_initialization_seed 或显式 seed。")
            effective_initialization_policy = "seeded"
        elif initialization_policy == "runtime":
            construction_seed = None
            effective_initialization_policy = "runtime"
        else:
            # Compatibility for existing request producers: before the new
            # fields are wired, seed continues to control construction too.
            construction_seed = seed
            effective_initialization_policy = "seeded" if seed is not None else "runtime"
        model_ref, params = request.get("model"), request.get("model_params", {})
        injections = runtime.validate_model_injections(
            request.get("model_injections", {})
        )
    if base_rng_devices is not None and not reset_rng:
        raise UserError("VARIANT_BASE_RNG_RESET_DISABLED", "该重建路径无法应用变体记录的基础模型随机设备。")
    if not model_ref:
        raise UserError(
            "MODEL_REQUIRED",
            "只需模型入口即可查看结构。",
            "使用 --model package.module:Model；无需训练配置。",
        )
    if not Path(root).is_dir():
        raise UserError("PROJECT_ROOT_MISSING", f"模型来源目录不存在：{root}")
    runtime.verify_model_injection_references(injections, root)
    code_root = request.get("code_snapshot_root") or root
    code_root = str(Path(code_root).resolve())
    project_root = str(Path(root).resolve())
    if code_root in sys.path:
        sys.path.remove(code_root)
    sys.path.insert(0, code_root)
    # Snapshot runs must not deliberately fall back to live project sources.
    configure_project_import_roots(request)
    if reset_rng and construction_seed is not None:
        random.seed(construction_seed)
        if base_rng_devices is None:
            runtime.torch.manual_seed(construction_seed)
    try:
        rng_scope = (layer_rng_scope(runtime.torch, construction_seed, base_rng_devices)
                     if reset_rng and construction_seed is not None and base_rng_devices is not None
                     else nullcontext())
        with rng_scope:
            factory = runtime.entrypoint(model_ref)
            model, injection_evidence = runtime.construct_model(
                factory, params, root, injections
            )
    except UserError:
        raise
    except Exception as error:
        raise UserError("MODEL_CONSTRUCTION_FAILED", str(error)) from error
    if not isinstance(model, runtime.torch.nn.Module):
        raise UserError("MODEL_TYPE_INVALID", "模型入口必须返回原生 torch.nn.Module。")
    # A legacy starter can accept **kwargs yet ignore them. Do not report an
    # assembly candidate as successful merely because its factory accepted the
    # new options; inspect the starter's declared construction state as well.
    assembly_options = {"input_routes", "encoder_output_modes"} & params.keys()
    if assembly_options:
        try:
            source = inspect.getsourcefile(factory)
            source_path = Path(source).resolve() if source else None
            is_starter = False
            if source_path and source_path.is_relative_to(Path(code_root)):
                with source_path.open(encoding="utf-8") as source_file:
                    is_starter = source_file.read(64).startswith("# Editable Model-Maven starter.")
        except (OSError, TypeError, ValueError, UnicodeError):
            is_starter = False
        if is_starter:
            values = vars(model)
            routes = values.get("input_routes")
            modes = values.get("native_encoder_outputs")
            missing = []
            expected_routes = params.get("input_routes")
            if isinstance(expected_routes, dict) and (
                not isinstance(routes, dict)
                or any(routes.get(slot) != source for slot, source in expected_routes.items())
            ):
                missing.append("input_routes")
            expected_modes = params.get("encoder_output_modes")
            if isinstance(expected_modes, dict) and (
                not isinstance(modes, (list, tuple, set))
                or any((slot in modes) != (mode == "native") for slot, mode in expected_modes.items())
            ):
                missing.append("encoder_output_modes")
            if missing:
                raise UserError(
                    "MODEL_ASSEMBLY_OPTIONS_UNCONFIRMED",
                    f"骨架接受了装配参数，但构造结果未确认：{', '.join(missing)}。",
                    "让模型应用对应接线和编码器输出配置，或在新目录创建新版骨架后迁移自定义代码；仅接受 **kwargs 不代表配置已生效。",
                    {"unconfirmed_options": missing},
                )
    # Training without an edit variant will place the factory result directly
    # on the selected training device later. Variant reconstruction and edit
    # history still have a CPU placement contract that must remain stable.
    source_edit = (request.get("operation") == "edit"
                   and isinstance(request.get("edit"), dict)
                   and request["edit"].get("target") == "source")
    if ((variant and device_policy == "legacy_cpu")
            or (not variant and device_policy == "legacy_cpu" and request.get("operation") not in observation_operations | {"training_probe", "train"} and not source_edit)):
        model.cpu()
    effects = []
    if variant:
        for edit in variant["edits"][: variant["cursor"]]:
            runtime.verify_component_references(edit, root)
            if device_policy == "factory":
                effects.extend(runtime.apply_history_item(model, edit, preserve_factory_device=True))
            else:
                effects.extend(runtime.apply_history_item(model, edit))
        # Detect new or modified local dependencies after imports.
        runtime.verify_manifest(variant["source_manifest"])
        if request.get("operation") != "edit" and request.get("seed") is not None:
            # The variant seed reconstructs model/edit initialization. An explicit
            # execution seed starts stochastic probe/training after reconstruction.
            seed = request["seed"]
            random.seed(seed)
            runtime.torch.manual_seed(seed)
    elif reset_rng and initialization_policy is not None and request.get("seed") is not None:
        # With an explicit split policy, restart the execution RNG only after
        # model construction so the two user-selected scopes remain distinct.
        seed = request["seed"]
        random.seed(seed)
        runtime.torch.manual_seed(seed)
    declarations = _declared_route_evidence(model)
    origin = {
        "model": model_ref,
        "params": params,
        "model_injections": injections,
        "injection_evidence": injection_evidence,
        **({"declared_routes": declarations} if declarations is not None else {}),
        "project_root": str(Path(root).resolve()),
    }
    if not variant and initialization_policy is not None:
        origin["model_initialization"] = {
            "policy": effective_initialization_policy,
            "seed_decimal": str(construction_seed) if construction_seed is not None else None,
            "scope": "python_random_and_torch_manual_seed_before_factory",
            "execution_seed_decimal": str(request["seed"]) if type(request.get("seed")) is int else None,
        }
    return model, origin, seed, effects
