"""Re-run one observed module invocation against an ephemeral copied boundary."""

import copy
import traceback


def _scope_evidence(module_path):
    return {
        "requested": "targeted",
        "executed": "selected-module-replay",
        "selected_module": module_path,
        "prerequisite": "a full root-model forward is executed first to observe and capture this invocation",
        "weights": "the selected module is deep-copied with its currently loaded state; this does not load or validate checkpoint files",
        "not_covered": [
            "root-model paths beyond the prerequisite forward",
            "backward propagation or parameter updates",
            "training behavior",
            "external side effects of user code",
        ],
        "cost": "not measured; model construction, selected runtime, a full root-model forward, a deep-copied selected module, and cloned invocation inputs are involved",
        "heavy_dependencies": {
            "status": "unknown",
            "reason": "the replay helper does not inventory project imports or runtime dependencies",
        },
        "resources": {
            "status": "not_measured",
            "steps": ["model construction", "full root-model forward", "selected-module copy and replay"],
            "additional_memory": ["deep-copied selected module", "cloned invocation inputs"],
        },
    }


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
            "execution_scope": _scope_evidence(invocation["selected_path"]),
            "isolation": "module copy and cloned input tensors; arbitrary global side effects are not isolated",
            "coverage": "先执行完整根模型前向以捕获调用，再对选中模块副本局部重放；不代表完整模型验证或训练通过。",
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
            "execution_scope": _scope_evidence(invocation["selected_path"]),
            "error": f"{type(error).__name__}: {error}"[:1000],
            "traceback": traceback.format_exc(limit=8)[-4000:],
            "coverage": "局部重放失败；此前完整根模型前向已执行，且其结果不会被改写。",
        }
