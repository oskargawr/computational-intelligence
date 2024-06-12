# mujoco

import gymnasium as gym
import numpy as np

env = gym.make("Ant-v4", render_mode="human")
observation, info = env.reset(seed=42)

# Definiowanie prostej strategii
actions = []

# Przykładowa strategia: naprzemiennie poruszaj każdą z nóg do przodu
for _ in range(150):
    actions.append(
        np.array([1, -1, 1, -1, 1, -1, 1, -1])
    )  # Porusz naprzemiennie nogami przez 150 kroków
for _ in range(150):
    actions.append(
        np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    )  # Porusz naprzemiennie nogami w przeciwnym kierunku przez 150 kroków
for _ in range(300):
    actions.append(np.array([0, 0, 0, 0, 0, 0, 0, 0]))  # Nie rób nic przez 300 kroków

for i in range(600):
    action = actions[i % len(actions)]
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)

env.close()

# odp c
# Action Space
# Box(-1.0, 1.0, (8,), float32)
# Observation Space
# Box(-inf, inf, (27,), float64)
