import gymnasium as gym
import numpy as np

env = gym.make("CarRacing-v2", render_mode="human")
observation, info = env.reset(seed=42)

# Definiowanie prostej strategii
# Przykładowo: przyspieszaj i utrzymuj stały kąt skrętu
actions = []

for _ in range(100):
    actions.append(np.array([0, 1, 0])) 
for _ in range(70):
    actions.append(np.array([-1, 0.6, 0.2]))
for _ in range(70):
    actions.append(np.array([0, 1, 0]))
for _ in range(3):
    actions.append(np.array([0.3, 0.8, 0.2]))

for i in range(600):
    action = actions[i % len(actions)]
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)

env.close()
