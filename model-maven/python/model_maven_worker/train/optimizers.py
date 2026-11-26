"""Construct explicitly selected optimizers for managed training.

This module does not choose a user's optimizer or install dependencies. The
caller owns request validation, provenance, checkpoint compatibility and the
decision to execute project code.
"""

import hashlib
import json

from ..protocol import UserError


def construct_optimizer(runtime, model, factory_ref, parameters):
    """Call a user factory with the model and validate its result.

    The optimizer may select a subset of trainable parameters, but may not
    own external parameters or include one parameter more than once. The
    caller must surface partial coverage rather than claiming a full update.
    """
    if not isinstance(factory_ref, str) or not factory_ref or len(factory_ref) > 1024:
        raise UserError("OPTIMIZER_FACTORY_INVALID", "优化器工厂必须是 module:callable。")
    if not isinstance(parameters, dict) or len(parameters) > 64:
        raise UserError("OPTIMIZER_PARAMETERS_INVALID", "优化器参数必须是最多 64 项的 JSON 对象。")
    if any(not isinstance(key, str) or not key.isidentifier() for key in parameters):
        raise UserError("OPTIMIZER_PARAMETERS_INVALID", "优化器参数名必须是 Python 标识符。")
    try:
        encoded = json.dumps(parameters, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as error:
        raise UserError("OPTIMIZER_PARAMETERS_INVALID", "优化器参数必须是有限、可序列化的 JSON。") from error
    if len(encoded.encode("utf-8")) > 65536:
        raise UserError("OPTIMIZER_PARAMETERS_INVALID", "优化器参数不能超过 64 KiB。")
    selected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not selected:
        raise UserError("MODEL_PARAMETERS_MISSING", "模型没有可训练参数。")
    try:
        factory = runtime.entrypoint(factory_ref)
        optimizer = factory(model, **parameters)
    except UserError:
        raise
    except Exception as error:
        raise UserError("OPTIMIZER_FACTORY_FAILED", f"优化器工厂执行失败：{type(error).__name__}: {error}") from error
    if not isinstance(optimizer, runtime.torch.optim.Optimizer):
        raise UserError("OPTIMIZER_FACTORY_INVALID", "工厂必须返回 torch.optim.Optimizer 实例。")
    try:
        owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    except (KeyError, TypeError) as error:
        raise UserError("OPTIMIZER_PARAMETERS_INVALID", "优化器参数组格式无效。") from error
    selected_ids = {id(parameter) for parameter in selected}
    owned_ids = [id(parameter) for parameter in owned]
    if not owned_ids or len(owned_ids) != len(set(owned_ids)) or not set(owned_ids).issubset(selected_ids):
        raise UserError("OPTIMIZER_PARAMETERS_INVALID", "受管优化器必须使用至少一个模型可训练参数，不能重复或包含其他参数。")
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    if not set(owned_ids).issubset(names):
        raise UserError("OPTIMIZER_PARAMETERS_INVALID", "优化器工厂改变了模型参数；请在构建模型时完成结构修改。")
    groups = [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]
    group_bytes = json.dumps(groups, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    cls = type(optimizer)
    return optimizer, {
        "factory": factory_ref,
        "parameters_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "optimizer_type": f"{cls.__module__}.{cls.__qualname__}",
        "parameter_count": len(selected),
        "optimized_parameter_count": len(owned_ids),
        "omitted_trainable_parameter_count": len(selected_ids - set(owned_ids)),
        "parameter_groups_sha256": hashlib.sha256(group_bytes).hexdigest(),
        "parameter_group_sizes": [len(group) for group in groups],
    }
