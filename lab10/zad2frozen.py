# Chromosomy - lista akcji (0, 1, 2, 3) - ruchy w lewo, prawo, góra, dół
# Fitness - suma odległości od celu (7, 7) w każdym ruchu
import gymnasium as gym
import numpy as np
import pygad
import random

env = gym.make(
    "FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="rgb_array"
)

actions = [0, 1, 2, 3]
goal_position = (7, 7)


def manhattan_distance(position, goal):
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def get_position(state, grid_size=8):
    return (state // grid_size, state % grid_size)


def fitness_func(ga_instance, solution, solution_idx):
    state, _ = env.reset()
    total_reward = 0
    for action in solution:
        state, reward, done, _, _ = env.step(action)
        current_position = get_position(state)

        if done and current_position == goal_position:
            return 0

        distance = manhattan_distance(current_position, goal_position)
        total_reward -= distance

    return total_reward


def on_start(ga_instance):
    population = []
    for _ in range(ga_instance.sol_per_pop):
        solution = [random.choice(actions) for _ in range(ga_instance.num_genes)]
        population.append(solution)
    ga_instance.population = np.array(population)


ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=100,
    init_range_low=0,
    init_range_high=3,
    mutation_percent_genes=10,
    on_start=on_start,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    gene_type=int,
    save_solutions=True,
    suppress_warnings=True,
)
ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()
print("Best solution:", solution)
print("Best solution fitness:", solution_fitness)
env.close()
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")
state, _ = env.reset()
for action in solution:
    state, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        break
env.close()
