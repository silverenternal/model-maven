"""Reconstruct a model from project inputs and replay a validated edit variant."""

import inspect
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError
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
    variant = request.get("variant")
    if variant:
        runtime.validate_variant(variant)
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
        model_ref, params = origin["model"], origin["params"]
        injections = origin_injections
    else:
        root = request["project_root"]
        seed = request.get("seed") if request.get("seed") is not None else 42
        model_ref, params = request.get("model"), request.get("model_params", {})
        injections = runtime.validate_model_injections(
            request.get("model_injections", {})
        )
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
    if project_root != code_root and project_root not in sys.path:
        sys.path.insert(1, project_root)
    if reset_rng:
        random.seed(seed)
        runtime.torch.manual_seed(seed)
    try:
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
    model.cpu()
    effects = []
    if variant:
        for edit in variant["edits"][: variant["cursor"]]:
            runtime.verify_component_references(edit, root)
            effects.extend(runtime.apply_history_item(model, edit))
        # Detect new or modified local dependencies after imports.
        runtime.verify_manifest(variant["source_manifest"])
        if request.get("operation") != "edit" and request.get("seed") is not None:
            # The variant seed reconstructs model/edit initialization. An explicit
            # execution seed starts stochastic probe/training after reconstruction.
            seed = request["seed"]
            random.seed(seed)
            runtime.torch.manual_seed(seed)
    declarations = _declared_route_evidence(model)
    return model, {
        "model": model_ref,
        "params": params,
        "model_injections": injections,
        "injection_evidence": injection_evidence,
        **({"declared_routes": declarations} if declarations is not None else {}),
        "project_root": str(Path(root).resolve()),
    }, seed, effects
