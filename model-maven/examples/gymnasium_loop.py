"""Optional Gymnasium example: Q-learning with episode-boundary recovery.

Install Gymnasium in your chosen Python environment before running. This is an
example, not an automatic Gymnasium adapter. The JSON checkpoint contains the
Q table and episode number; it does not serialize an environment mid-episode.
"""

import math
import random

import gymnasium as gym


def run(context, *, seed=41, learning_rate=0.2):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError("learning_rate must be a number")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be finite and between 0 and 1")

    saved = context.resume_state
    values = saved["q_values"] if saved is not None else [[0.0] * 4 for _ in range(16)]
    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        while context.remaining_steps:
            episode = context.completed_steps + 1
            observation, _ = env.reset(seed=seed + episode)
            rng = random.Random(seed + episode)
            reward = 0.0
            for _ in range(100):
                action = (rng.randrange(4) if rng.random() < 0.25
                          else max(range(4), key=lambda i: values[int(observation)][i]))
                next_observation, reward, terminated, truncated, _ = env.step(action)
                old = values[int(observation)][action]
                target = reward + (0 if terminated else 0.9 * max(values[int(next_observation)]))
                values[int(observation)][action] = old + learning_rate * (target - old)
                observation = next_observation
                if terminated or truncated:
                    break
            if not context.report_step({"reward": float(reward)}, state={"q_values": values, "episode": episode}):
                break
        return {"episodes": context.completed_steps}
    finally:
        env.close()
