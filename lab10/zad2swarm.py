# Fitness - suma nagród z gry

import gymnasium as gym
import numpy as np
import pyswarms as ps


env_name = "LunarLander-v2"
env = gym.make(env_name, render_mode="rgb_array")


action_space_size = env.action_space.n
state_space_size = env.observation_space.shape[0]


def fitness_func(solution):
    fitness_scores = np.zeros(solution.shape[0])

    for i, particle in enumerate(solution):
        state, _ = env.reset()
        total_reward = 0

        for action in particle:
            state, reward, done, _, _ = env.step(int(action))
            total_reward += reward

            if done:
                break

        fitness_scores[i] = total_reward

    return -fitness_scores


options = {"c1": 0.5, "c2": 0.3, "w": 0.9}


num_particles = 50
num_iterations = 10
num_genes = 100


bounds = (np.zeros(num_genes), (action_space_size - 1) * np.ones(num_genes))


optimizer = ps.single.GlobalBestPSO(
    n_particles=num_particles, dimensions=num_genes, options=options, bounds=bounds
)


best_cost, best_pos = optimizer.optimize(fitness_func, iters=num_iterations)


print("Best solution:", best_pos)
print("Best solution fitness:", -best_cost)
env.close()
env = gym.make(env_name, render_mode="human")
state, _ = env.reset()
env.render()
for action in best_pos:
    state, reward, done, _, _ = env.step(int(action))
    env.render()
    if done:
        break

env.close()
