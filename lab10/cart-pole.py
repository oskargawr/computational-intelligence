# classic control environment

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(600):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
  
    if terminated or truncated:
        observation, info = env.reset(seed=42)

env.close()

# odp b
# Action Space
# Discrete(2)
# Observation Space
# Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)