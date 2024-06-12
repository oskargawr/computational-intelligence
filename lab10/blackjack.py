# toy text

import gymnasium as gym

env = gym.make("Blackjack-v1", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(600):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)

env.close()

# odp a
# Action Space
# Discrete(2)
# Observation Space
# Tuple(Discrete(32), Discrete(11), Discrete(2))
