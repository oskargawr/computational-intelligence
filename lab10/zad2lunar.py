import gymnasium as gym
import pygad
import numpy as np

# Utworzenie środowiska
env = gym.make("LunarLander-v2")

# Parametry genetyczne
sol_per_pop = 50
num_genes = 300


# Definiowanie funkcji fitness
def fitness_func(model, solution, solution_idx):
    observation, info = env.reset(seed=42)
    total_reward = 0
    for action in solution:
        observation, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


# Parametry algorytmu genetycznego
ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=[0, 1, 2, 3],
    mutation_percent_genes=10,
)

# Uruchomienie algorytmu
ga_instance.run()

# Pobranie najlepszego rozwiązania
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Najlepsze rozwiązanie: {solution}, Fitness: {solution_fitness}")

# Testowanie najlepszego rozwiązania
# observation, info = env.reset(seed=42)
env = gym.make("LunarLander-v2", render_mode="human")
# env.render()
state, _ = env.reset()
for action in solution:
    observation, reward, terminated, truncated, info = env.step(int(action))
    env.render()
    if terminated or truncated:
        break
env.close()

# Fitness: 141.48304203239553
