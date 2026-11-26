"""A tiny deterministic bandit illustrating a framework-neutral user loop.

This toy environment is deliberately simple; replace it with your own
environment and learning library. Model-Maven records the state you report,
not unreported environment state, random generators or external data.
"""

import math


def run(context, *, learning_rate=0.2):
    """Use with ``loop-run --loop-interface context --loop bandit_loop:run``."""
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError("learning_rate must be a number")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be finite and between 0 and 1")

    state = context.resume_state if context.resume_state is not None else {
        "values": [0.0, 0.0], "steps": 0,
    }
    values = list(state["values"])
    while context.remaining_steps:
        step = context.completed_steps + 1
        action = step % 2
        reward = 1.0 if action == 1 else 0.0
        values[action] += learning_rate * (reward - values[action])
        state = {"values": values, "steps": step}
        if not context.report_step({"reward": reward, "estimate": values[action]}, state=state):
            break
    return {"final_values": values}
