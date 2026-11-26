"""Framework-neutral resumable loop; replace the body with your own library."""


def run(_model, context, *, increment=1, checkpoint_every=1):
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    value = context.resume_state["value"] if context.resume_state is not None else 0
    while context.remaining_steps:
        value += increment
        # Frequent state reports cost source rechecks and durable disk writes.
        # An interruption can lose progress since the last reported state.
        step = context.completed_steps + 1
        save = step % checkpoint_every == 0 or context.remaining_steps == 1
        state = {"value": value} if save else None
        if not context.report_step({"value": value}, state=state):
            break
    return {"final_value": value}


def run_context(context, *, increment=1, checkpoint_every=1):
    """Use with --loop-interface context; no placeholder model is needed."""
    return run(None, context, increment=increment, checkpoint_every=checkpoint_every)
