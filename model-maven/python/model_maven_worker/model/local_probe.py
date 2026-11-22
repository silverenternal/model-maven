"""Re-run one observed module invocation against an ephemeral copied boundary."""

import copy
import traceback


def _clone_input(torch_module, value):
    if isinstance(value, torch_module.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_input(torch_module, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_input(torch_module, item) for item in value)
    if isinstance(value, list):
        return [_clone_input(torch_module, item) for item in value]
    return copy.deepcopy(value)


def replay_invocation(module, call_args, call_kwargs, invocation, *, torch_module, tree_fn):
    """Replay a captured call on a copied module and cloned inputs; persist no tensor data."""
    try:
        candidate = copy.deepcopy(module)
        args = _clone_input(torch_module, call_args)
        kwargs = _clone_input(torch_module, call_kwargs)
        candidate.train(module.training)
        with torch_module.no_grad():
            output = candidate(*args, **kwargs)
        return {
            "status": "observed-runtime",
            "module_path": invocation["selected_path"],
            "invocation_id": invocation["invocation_id"],
            "invocation_index": invocation["invocation_index"],
            "probe_id": invocation["probe_id"],
            "forward_id": invocation["forward_id"],
            "batch_id": invocation["batch_id"],
            "input_signature": invocation["input_signature"],
            "inputs": tree_fn(torch_module, {"args": args, "kwargs": kwargs}),
            "outputs": tree_fn(torch_module, output),
            "isolation": "module copy and cloned input tensors; arbitrary global side effects are not isolated",
            "coverage": "仅重复一个已观察调用的局部模块；不代表根模型其余路径或训练更新。",
        }
    except Exception as error:
        return {
            "status": "failed",
            "module_path": invocation["selected_path"],
            "invocation_id": invocation["invocation_id"],
            "invocation_index": invocation["invocation_index"],
            "probe_id": invocation["probe_id"],
            "forward_id": invocation["forward_id"],
            "batch_id": invocation["batch_id"],
            "input_signature": invocation["input_signature"],
            "error": f"{type(error).__name__}: {error}"[:1000],
            "traceback": traceback.format_exc(limit=8)[-4000:],
            "coverage": "局部重放失败；根模型原次前向结果不会被改写。",
        }
