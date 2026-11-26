"""Explicit split-specific transforms around a user-owned batch factory.

This adapter neither discovers data nor changes split arguments for the source.
The source factory must itself return a batch, not a Dataset object. Each call
uses the declared seed; callers should select it deliberately for stochastic
transforms and account for any separately configured project augmentation.
"""
from .augment import apply_batch_augmentations
from ..protocol import UserError
from ..runtime.entrypoints import entrypoint


def batch(*, factory, factory_params=None, split, split_transforms, seed):
    if not isinstance(factory, str) or factory == "model_maven_worker.data.pipeline:batch":
        raise UserError("GRAPH_PIPELINE_FACTORY_INVALID", "请选择一个非递归的批次工厂。")
    if factory_params is None:
        factory_params = {}
    if not isinstance(factory_params, dict):
        raise UserError("GRAPH_PIPELINE_FACTORY_INVALID", "批次工厂参数必须是对象。")
    if not isinstance(split, str) or not split or not isinstance(split_transforms, dict):
        raise UserError("GRAPH_PIPELINE_SPLIT_INVALID", "必须明确选择变换切分和切分流水线。")
    if split not in split_transforms or not isinstance(split_transforms[split], list):
        raise UserError("GRAPH_PIPELINE_SPLIT_INVALID", "当前切分没有有序变换列表。")
    if len(split_transforms) > 32 or len(split_transforms[split]) > 64:
        raise UserError("GRAPH_PIPELINE_SPLIT_INVALID", "切分或变换步骤超过限制。")
    if type(seed) is not int or not 0 <= seed <= 2**53 - 1:
        raise UserError("GRAPH_PIPELINE_SEED_INVALID", "流水线种子必须是可精确表示的非负整数。")
    value = entrypoint(factory)(**factory_params)
    # An explicitly recognized PyTorch Dataset is not a batch. Do not infer
    # that arbitrary iterable objects (for example NumPy arrays) are datasets.
    import sys
    torch = sys.modules.get("torch")
    dataset_type = getattr(getattr(getattr(torch, "utils", None), "data", None), "Dataset", None)
    if isinstance(dataset_type, type) and isinstance(value, dataset_type):
        raise UserError("GRAPH_PIPELINE_NEEDS_BATCH", "变换包装器需要批次；数据集请先通过用户工厂组批。")
    transformed, _observations = apply_batch_augmentations(value, split_transforms[split], resolve_entrypoint=entrypoint, seed=seed)
    return transformed
