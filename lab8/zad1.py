import pygad
import numpy 
import matplotlib.pyplot as plt
import time

# Lista przedmiotów [wartość, waga, nazwa]
items = [
    [100, 7, "komputer"],
    [300, 7, "laptop"],
    [200, 6, "torba"],
    [40, 2, "dlugopis"],
    [500, 5, "kosiarka"],
    [70, 6, "wentylator"],
    [100, 1, "bankomat"],
    [250, 3, "samolot"],
    [300, 10, "jacht"],
    [280, 3, "odkurzacz"],
]

max_weight = 25  # Maksymalna waga


# Funkcja fitness
def fitness_func(model, solution, solution_idx):
    total_value = numpy.sum(
        numpy.array(solution) * numpy.array([item[0] for item in items])
    )
    total_weight = numpy.sum(
        numpy.array(solution) * numpy.array([item[1] for item in items])
    )
    if total_weight > max_weight:
        fitness = 0  # Rozwiązanie przekracza maksymalną wagę
    else:
        fitness = total_value
    return fitness


# Parametry algorytmu genetycznego
sol_per_pop = 10 # wielkosc populacji
num_genes = len(items) # liczba genów
num_parents_mating = 5 # liczba rodzicow
num_generations = 30 # liczba generacji
keep_parents = 2 # liczba rodzicow do zachowania
gene_space = [0, 1] # Przestrzeń genów
parent_selection_type = "sss" # typ selekcji rodzicow
crossover_type = "single_point" # typ krzyzowania
mutation_type = "random"
mutation_percent_genes = 10  # procent genów do mutacji

# Zadanie części e: Ile razy algorytm znajduje najlepsze rozwiązanie (wartość 1630)
correct_solutions = 0
num_runs = 10

for _ in range(num_runs):
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
        stop_criteria="reach_1630",
        save_best_solutions=True,
    )
    ga_instance.run()

    solution = ga_instance.best_solutions[-1]
    solution_fitness = numpy.sum(
        numpy.array(solution) * numpy.array([item[0] for item in items])
    )
    total_value = solution_fitness
    if total_value == 1630:
        correct_solutions += 1

accuracy = (correct_solutions / num_runs) * 100
print(f"Accuracy of finding the best solution: {accuracy:.2f}%")

# Zadanie części f: Średni czas działania algorytmu przy znalezieniu najlepszego rozwiązania
correct_solutions = 0
total_time = 0

while correct_solutions < 10:
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
        stop_criteria="reach_1630",
        save_best_solutions=True,
    )
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()

    solution = ga_instance.best_solutions[-1]
    solution_fitness = numpy.sum(
        numpy.array(solution) * numpy.array([item[0] for item in items])
    )
    total_value = solution_fitness

    if total_value == 1630:
        correct_solutions += 1
        total_time += end_time - start_time

average_time = total_time / correct_solutions
print(f"Average time to find the best solution: {average_time:.4f} seconds")

# Wyniki dla najlepszego rozwiązania
solution = ga_instance.best_solutions[-1]
solution_fitness = numpy.sum(
        numpy.array(solution) * numpy.array([item[0] for item in items])
    )
selected_items = [items[i][2] for i in range(len(solution)) if solution[i] == 1]
print("Parameters of the best solution: ", solution)
print("Fitness value of the best solution: ", solution_fitness)
print("Selected items: ", selected_items)
print("Total value of the best solution: ", solution_fitness)

# Wykres optymalizacji
ga_instance.plot_fitness()


# Accuracy of finding the best solution: 100.00%
# Average time to find the best solution: 0.0029 seconds
# Parameters of the best solution:  [0. 1. 1. 0. 1. 0. 1. 1. 0. 1.]
# Fitness value of the best solution:  1630.0
# Selected items:  ['laptop', 'torba', 'kosiarka', 'bankomat', 'samolot', 'odkurzacz']
# Total value of the best solution:  1630.0