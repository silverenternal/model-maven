

def step(model, batch, context, objectives=None):
    """Managed-step adapter; objectives are explicit and may be edited here.

    Example: [{"output": "scores", "target": "label", "loss": "cross_entropy"}]
    Batch inputs use args/kwargs and named targets, or one unnamed target when
    every objective refers to the same target name.
    """
    import math

    if objectives is None:
        objectives = GRAPH_OBJECTIVES
    if not isinstance(objectives, list) or not 1 <= len(objectives) <= 64:
        raise ValueError("objectives must be a nonempty list of at most 64 entries")
    if not isinstance(batch, dict) or not isinstance(batch.get("kwargs"), dict) or not isinstance(batch.get("args", []), (list, tuple)):
        raise ValueError("batch must provide args and a kwargs mapping")
    loss_inputs = batch.get("loss_inputs")
    targets = loss_inputs.get("targets", loss_inputs.get("target")) if isinstance(loss_inputs, dict) else None
    if targets is not None and not isinstance(targets, dict):
        names = set()
        for objective in objectives:
            if not isinstance(objective, dict):
                raise ValueError("objectives must contain objects")
            name = objective.get("target", objective.get("output"))
            if not isinstance(name, str):
                raise ValueError("each objective must name its target")
            names.add(name)
        if len(names) != 1:
            raise ValueError("an unnamed target requires all objectives to use one target name")
        targets = {next(iter(names)): targets}
    if not isinstance(targets, dict):
        raise ValueError("batch.loss_inputs must provide targets")
    predictions = model(*batch.get("args", []), **batch["kwargs"])
    if not isinstance(predictions, dict):
        raise ValueError("graph model must return a mapping of output node IDs")
    losses = {}
    total = None
    for index, objective in enumerate(objectives):
        if not isinstance(objective, dict) or set(objective) - {"output", "target", "loss", "weight"}:
            raise ValueError(f"objective {index} has invalid fields")
        output = objective.get("output")
        target = objective.get("target", output)
        kind = objective.get("loss")
        weight = objective.get("weight", 1.0)
        if not isinstance(output, str) or not isinstance(target, str) or output not in predictions or target not in targets:
            raise ValueError(f"objective {index} refers to a missing output or target")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"objective {index} needs a finite nonnegative weight")
        prediction, expected = predictions[output], targets[target]
        if kind == "mse":
            if not isinstance(prediction, torch.Tensor) or not isinstance(expected, torch.Tensor) or prediction.shape != expected.shape:
                raise ValueError(f"objective {index} ({output}): mse requires equal tensor shapes")
            loss = nn.functional.mse_loss(prediction, expected)
        elif kind == "cross_entropy":
            loss = nn.functional.cross_entropy(prediction, expected)
        elif isinstance(kind, str) and kind.count(":") == 1:
            module, name = kind.split(":", 1)
            if not module or not name:
                raise ValueError(f"objective {index} has an invalid custom loss reference")
            function = importlib.import_module(module)
            for part in name.split("."):
                function = getattr(function, part)
            loss = function(prediction, expected)
        else:
            raise ValueError(f"objective {index} needs mse, cross_entropy, or a module:callable loss")
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError(f"objective {index} must produce a scalar tensor")
        losses[f"objective_{index}"] = loss
        weighted = weight * loss
        total = weighted if total is None else total + weighted
    return {"loss": total, "losses": losses}
