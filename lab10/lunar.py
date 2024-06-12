import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

# Definiowanie prostej strategii
actions = []

for _ in range(8):
    actions.append(2)  # Użyj głównego silnika 
for _ in range(14):
    actions.append(1)  # Użyj lewego bocznego silnika 
for _ in range(14):
    actions.append(3)  # Użyj prawego bocznego 
for _ in range(10):
    actions.append(0)  # Nie rób nic

for i in range(600):
    action = actions[i % len(actions)]
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)

env.close()
