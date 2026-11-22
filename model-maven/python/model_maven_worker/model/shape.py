"""Best-effort FX/Meta shape inference with explicit unsupported outcomes."""

import copy
import re


def _meta_value(torch_module, value):
    if isinstance(value, torch_module.Tensor):
        return value.detach().to(device="meta")
    if isinstance(value, dict):
        return {key: _meta_value(torch_module, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_meta_value(torch_module, item) for item in value)
    if isinstance(value, list):
        return [_meta_value(torch_module, item) for item in value]
    return value


def _tensor_metadata(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": [int(size) for size in value.shape], "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return [_tensor_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _tensor_metadata(item) for key, item in value.items()}
    return None


def infer_with_fx_meta(model, args, kwargs, *, torch_module):
    """Try FX tracing plus Meta execution; tracing failures never become contract failures."""
    try:
        candidate = copy.deepcopy(model).to(device="meta")
        was_training = model.training
        candidate.train(was_training)
        graph_module = torch_module.fx.symbolic_trace(candidate)
        meta_args = _meta_value(torch_module, args)
        meta_kwargs = _meta_value(torch_module, kwargs)
        from torch.fx.passes.shape_prop import ShapeProp

        ShapeProp(graph_module).propagate(*meta_args, **meta_kwargs)
        nodes = []
        for node in graph_module.graph.nodes:
            metadata = _tensor_metadata(node.meta.get("tensor_meta"))
            if metadata is not None:
                nodes.append({"node": node.name, "target": str(node.target), "value": metadata})
        return {
            "status": "inferred-static",
            "engine": "torch.fx+meta",
            "nodes": nodes,
            "coverage": "FX-traceable graph only; dynamic Python branches and untraced code are not covered.",
        }
    except Exception as error:
        message = str(error)
        detail = str(error.__cause__) if error.__cause__ else message
        node_match = re.search(r"node=%([^\s:]+)", message)
        node_path = node_match.group(1) if node_match else "model"
        mismatch = None
        reduction = re.search(r"reduction dim, but got \[\d+,\s*(\d+)\] X \[(\d+),", detail)
        if reduction:
            mismatch = {"expected": int(reduction.group(2)), "actual": int(reduction.group(1)),
                        "field": "linear.in_features"}
        else:
            matrix = re.search(r"shapes cannot be multiplied \(\d+x(\d+) and (\d+)x\d+\)", detail)
            if matrix:
                mismatch = {"expected": int(matrix.group(2)), "actual": int(matrix.group(1)),
                            "field": "matmul.inner_dimension"}
        if mismatch:
            return {
                "status": "failed",
                "engine": "torch.fx+meta",
                "diagnostics": [{"path": node_path, **mismatch}],
                "reason": message[:1000],
                "coverage": "Meta 推演已证实该 FX 节点的静态维度不匹配；没有执行真实前向。",
            }
        return {
            "status": "unsupported",
            "engine": "torch.fx+meta",
            "reason": f"{type(error).__name__}: {error}"[:1000],
            "coverage": "静态推演未能覆盖；需以真实批次 probe 观察已执行路径。",
        }
