import pygad
import numpy as np
import math


def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


gene_space = {"low": 0.0, "high": 1.0}

def fitness_func(ga_instance, solution, solution_idx):
    x, y, z, u, v, w = solution
    return endurance(x, y, z, u, v, w)

sol_per_pop = 50
num_genes = 6
num_parents_mating = 25
num_generations = 100
keep_parents = 5
parent_selection_type = "tournament"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 20  

ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print(
    "Fitness value of the best solution = {solution_fitness}".format(
        solution_fitness=solution_fitness
    )
)

# Wykres optymalizacji
ga_instance.plot_fitness()

# Parameters of the best solution : [0.87288293 0.76658476 0.99661113 0.99907261 0.01463581 0.00980532]
# Fitness value of the best solution = 2.83913244631625