import numpy as np
import pygad
import matplotlib.pyplot as plt
import time

labirynt = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)


def fitness_func(model, solution, solution_idx):
    x, y = 1, 1
    penalty = 0
    award = 0

    for move_idx, move in enumerate(solution):
        if x == 10 and y == 10:
            award = 10 / (
                move_idx + 1
            )  # Nagroda za dojście do końca, im mniej kroków, tym większa nagroda
            break
        elif move == 0 and labirynt[x - 1, y] != 1:  # w górę
            x -= 1
        elif move == 1 and labirynt[x + 1, y] != 1:  # w dół
            x += 1
        elif move == 2 and labirynt[x, y - 1] != 1:  # w lewo
            y -= 1
        elif move == 3 and labirynt[x, y + 1] != 1:  # w prawo
            y += 1
        else:
            penalty += (abs(10 - x) + abs(10 - y)) * 0.5  # Kara za nielegalny ruch
            break

    distance_to_exit = abs(10 - x) + abs(10 - y)
    fitness = 1 / (distance_to_exit + 1) - penalty + award

    return fitness


num_generations = 5000
sol_per_pop = 80  # Zwiększona populacja
num_genes = 30  # Każdy ruch jest reprezentowany przez jedną liczbę z zakresu 0-3, więc potrzebujemy 30 genów
num_parents_mating = 20  # Więcej rodziców do mieszania genów
keep_parents = 5  # Więcej rodziców do zachowania różnorodności
gene_space = [0, 1, 2, 3]  # Każdy gen to jedna z 4 liczb (0, 1, 2, 3)
mutation_percent_genes = (
    5  # Zmniejszony procent mutacji, aby uniknąć ostrzeżeń i nadmiernych mutacji
)

times = []
for i in range(10):
    print(i, "próba")
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        mutation_percent_genes=mutation_percent_genes,
        num_parents_mating=num_parents_mating,
        keep_parents=keep_parents,
        fitness_func=fitness_func,
        stop_criteria="reach_1",
    )
    start = time.time()
    ga_instance.run()
    end = time.time()
    times.append(end - start)

print("wszystkie czasy:")
for i in times:
    print(f"{i:.2f} s")
print(f"średni czas: {sum(times)/10:.2f}")

ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=1000,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    mutation_percent_genes=mutation_percent_genes,
    num_parents_mating=num_parents_mating,
    keep_parents=keep_parents,
    fitness_func=fitness_func,
)

start = time.time()
ga_instance.run()
end = time.time()

solution = ga_instance.best_solution()[0]
solution_fitness = fitness_func(ga_instance, solution, -1)
print("Najlepsze rozwiązanie:", solution)
print("Wartość funkcji fitness dla najlepszego rozwiązania:", solution_fitness)
directions_mapping = {0: "góra", 1: "dół", 2: "lewo", 3: "prawo"}

solution_words = [directions_mapping[step] for step in solution]

print("Najlepsze rozwiązanie:", solution_words)


def simulate_solution(labirynt, solution):
    x, y = 1, 1
    path = [(x, y)]

    for move in solution:
        if (x, y) == (10, 10):
            print("Sukces! Dotarliśmy do wyjścia.")
            return path
        elif move == 0 and labirynt[x - 1, y] != 1:  # góra
            x -= 1
        elif move == 1 and labirynt[x + 1, y] != 1:  # dół
            x += 1
        elif move == 2 and labirynt[x, y - 1] != 1:  # lewo
            y -= 1
        elif move == 3 and labirynt[x, y + 1] != 1:  # prawo
            y += 1
        else:
            print(f"Nielegalny ruch: ({x}, {y}) przy ruchu {directions_mapping[move]}")
            break
        path.append((x, y))

    if (x, y) == (10, 10):
        print("Algorytm znalazł ścieżkę")
    else:
        print("x,y", x, y)
        print("Nie udało się dotrzeć do wyjścia.")
    return path


# Przeprowadzenie symulacji
path = simulate_solution(labirynt, solution)
print("Ścieżka:", path, ", liczba kroków ", len(path) - 1)


# Wizualizacja labiryntu i najlepszej ścieżki
def plot_labirynt(labirynt, path):
    plt.figure(figsize=(10, 10))
    plt.imshow(labirynt, cmap="binary")

    # Start (S) na zielono
    plt.text(
        1,
        1,
        "S",
        ha="center",
        va="center",
        color="green",
        fontsize=12,
        fontweight="bold",
    )
    # Exit (E) na czerwono
    plt.text(
        10,
        10,
        "E",
        ha="center",
        va="center",
        color="red",
        fontsize=12,
        fontweight="bold",
    )

    # Ścieżka na niebiesko
    for x, y in path:
        plt.text(y, x, "x", ha="center", va="center", color="blue", fontsize=8)

    plt.gca().invert_yaxis()
    plt.show()


plot_labirynt(labirynt, path)
ga_instance.plot_fitness()
plt.savefig("fitness_plot.png")
print("=============== czas", end)
