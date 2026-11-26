"""User model entrypoints, declared seams and construction-site capture."""

import inspect
import re

from ..protocol import UserError
from .discovery import ConstructionSiteCollector, project_construction_site


def validate_model_injections(value):
    if not isinstance(value, dict) or len(value) > 32:
        raise UserError("MODEL_INJECTION_INVALID", "模型接缝必须是最多 32 项的对象。")
    injections = {}
    for slot, specification in value.items():
        if not isinstance(slot, str) or not slot.isidentifier():
            raise UserError("MODEL_INJECTION_INVALID", f"模型接缝名无效：{slot!r}")
        if isinstance(specification, dict):
            factory = specification.get("factory")
            params = specification.get("params", {})
            reference = specification.get("component_ref")
            if (not isinstance(params, dict) or set(specification) - {"factory", "params", "component_ref"}
                    or ("component_ref" in specification and not isinstance(reference, dict))):
                raise UserError(
                    "MODEL_INJECTION_INVALID",
                    f"接缝 {slot} 的工厂参数或组件锁格式无效。",
                )
            if reference is not None and (
                not isinstance(reference.get("id"), str)
                or not isinstance(reference.get("version"), str)
                or not isinstance(reference.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", reference.get("sha256", ""))
            ):
                raise UserError("MODEL_INJECTION_INVALID", f"接缝 {slot} 的组件版本锁格式无效。")
        else:
            factory = specification
            params = {}
            reference = None
        if not isinstance(factory, str) or factory.count(":") != 1:
            raise UserError("MODEL_INJECTION_INVALID", f"接缝 {slot} 的工厂入口须为 module:callable。")
        module_name, attribute = factory.split(":")
        if not all(part.isidentifier() for part in module_name.split(".")) or not all(
            part.isidentifier() for part in attribute.split(".")
        ):
            raise UserError("MODEL_INJECTION_INVALID", f"接缝 {slot} 的工厂入口格式无效：{factory}")
        if isinstance(specification, dict):
            injections[slot] = {"factory": factory, "params": params}
            if reference is not None:
                injections[slot]["component_ref"] = reference
        else:
            injections[slot] = factory
    return injections


def construct_model(factory, params, project_root, injections=None, *, torch_module, resolve_entrypoint, module_entries_fn):
    """Capture module creation sites during one factory call without retaining hooks."""
    source_sites = ConstructionSiteCollector(project_root)
    module_type = torch_module.nn.Module
    original_init = module_type.__init__

    def tracked_init(module, *args, **kwargs):
        original_init(module, *args, **kwargs)
        source_sites.record(module)

    injections = validate_model_injections(injections or {})
    if set(params) & set(injections):
        duplicate = sorted(set(params) & set(injections))[0]
        raise UserError(
            "MODEL_INJECTION_CONFLICT",
            f"模型参数和组件注入重复使用关键字 {duplicate}。",
            "移除同名 --model-param 或 --inject；一个接缝只能绑定一个来源。",
        )

    # Starter routing options were added after the first editable starter was
    # released. Detect a clearly incompatible factory before running component
    # constructors, while leaving dynamic **kwargs factories to validate their
    # own contract during the real candidate forward.
    assembly_options = {"input_routes", "encoder_output_modes"} & params.keys()
    if assembly_options:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            missing = sorted(
                name for name in assembly_options
                if name not in signature.parameters
                or signature.parameters[name].kind == inspect.Parameter.POSITIONAL_ONLY
            )
            if missing:
                raise UserError(
                    "MODEL_ASSEMBLY_OPTIONS_UNSUPPORTED",
                    f"模型工厂未声明装配参数：{', '.join(missing)}。",
                    "旧版自定义骨架需由作者适配相应工厂参数及模型接线逻辑；也可在新目录创建新版骨架并迁移自定义代码。",
                    {"unsupported_options": missing},
                )

    module_type.__init__ = tracked_init
    try:
        components = {}
        for slot, specification in injections.items():
            if isinstance(specification, dict):
                reference = specification["factory"]
                component_params = specification["params"]
            else:
                reference = specification
                component_params = {}
            component_factory = resolve_entrypoint(reference)
            component = component_factory(**component_params)
            if not isinstance(component, torch_module.nn.Module):
                raise UserError(
                    "MODEL_INJECTION_TYPE_INVALID",
                    f"接缝 {slot} 的工厂 {reference} 返回 {type(component).__name__}，需要 nn.Module。",
                )
            components[slot] = component
        if components:
            try:
                factory_signature = inspect.signature(factory)
            except (TypeError, ValueError):
                # Native and dynamically wrapped callables may be callable even
                # when Python cannot expose a signature. Let the actual call
                # decide instead of misreporting an undeclared seam.
                factory_signature = None
            if factory_signature is not None:
                try:
                    factory_signature.bind(**params, **components)
                except TypeError as error:
                    raise UserError(
                        "MODEL_INJECTION_SEAM_UNDECLARED",
                        f"根模型工厂没有声明所需的组件接缝：{error}",
                        "让工厂显式接受同名关键字参数，例如 build_model(encoder): return Model(encoder)。",
                    ) from error
        model = factory(**params, **components)
    finally:
        module_type.__init__ = original_init
    if isinstance(model, torch_module.nn.Module):
        source_locations, source_site_counts = source_sites.finish(
            model, module_entries_fn
        )
    else:
        source_locations, source_site_counts = {}, {}
        source_sites.discard()
    injection_evidence = []
    if injections:
        if not isinstance(model, torch_module.nn.Module):
            raise UserError("MODEL_TYPE_INVALID", "模型入口必须返回原生 torch.nn.Module。")
        _, aliases = module_entries_fn(model)
        unused = sorted(slot for slot, component in components.items() if id(component) not in aliases)
        if unused:
            raise UserError(
                "MODEL_INJECTION_UNUSED",
                f"根模型没有注册接缝注入的模块：{', '.join(unused)}。",
                "检查根工厂是否把每个注入模块接入返回模型；不能仅接收后忽略组件。",
            )
        for slot, component in components.items():
            specification = injections[slot]
            if isinstance(specification, dict):
                factory_ref = specification["factory"]
                recorded_params = specification["params"]
            else:
                factory_ref = specification
                recorded_params = {}
            injection_evidence.append({
                "slot": slot,
                "factory": factory_ref,
                "params": recorded_params,
                "module_type": type(component).__name__,
                "registered_paths": aliases[id(component)],
                "status": "registered",
                # Fact about this construction only. A factory may return a
                # different instance on the next reconstruction.
                "shared_with_slots": sorted(
                    other_slot for other_slot, other in components.items()
                    if other_slot != slot and other is component
                ),
            })
            if isinstance(specification, dict) and "component_ref" in specification:
                injection_evidence[-1]["component_ref"] = specification["component_ref"]
    return model, injection_evidence, source_locations, source_site_counts


