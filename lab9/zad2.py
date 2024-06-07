import matplotlib.pyplot as plt
import random
from aco import AntColony

plt.style.use("dark_background")


def generate_random_coords(n):
    return [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n)]


def plot_nodes(coords, ax):
    for x, y in coords:
        ax.plot(x, y, "g.", markersize=15)
    ax.axis("off")


def plot_optimal_path(coords, optimal_nodes, title, ax):
    for i in range(len(optimal_nodes) - 1):
        ax.plot(
            (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
            (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
        )
    ax.plot(
        (optimal_nodes[-1][0], optimal_nodes[0][0]),
        (optimal_nodes[-1][1], optimal_nodes[0][1]),
    )
    ax.set_title(title)


def run_experiment(
    coords,
    ant_count,
    alpha,
    beta,
    pheromone_evaporation_rate,
    pheromone_constant,
    iterations,
    ax,
):
    plot_nodes(coords, ax)

    colony = AntColony(
        coords,
        ant_count=ant_count,
        alpha=alpha,
        beta=beta,
        pheromone_evaporation_rate=pheromone_evaporation_rate,
        pheromone_constant=pheromone_constant,
        iterations=iterations,
    )

    optimal_nodes = colony.get_path()
    plot_optimal_path(
        coords,
        optimal_nodes,
        title=f"ant_count: {ant_count}, alpha: {alpha}, beta: {beta}, "
        f"evaporation_rate: {pheromone_evaporation_rate}, "
        f"constant: {pheromone_constant}, iterations: {iterations}",
        ax=ax,
    )

    print(f"Optimal path: {optimal_nodes}")
    print(
        f"Ant count: {ant_count}, alpha: {alpha}, beta: {beta}, pheromone evaporation rate: {pheromone_evaporation_rate}, pheromone constant: {pheromone_constant}, iterations: {iterations}"
    )
    print("\n")


coords = generate_random_coords(
    50
) 

# parametry to: liczba wierzcholkow, liczba mrowek, parametr alfa (im wyzszy tym wieksze znaczenie feromonow), parametr beta (im wyzszy tym wieksze znaczenie odleglosci), wspolczynnik parowania feromonow, stala feromonowa, liczba iteracji

experiments = [
    (10, 0.5, 1.2, 0.4, 1000.0, 30),
    (20, 0.7, 1.0, 0.3, 500.0, 50),
    (60, 0.9, 1.5, 0.2, 2000.0, 300),
    (100, 0.6, 1.1, 0.5, 1500.0, 200),
]

fig, axes = plt.subplots(len(experiments), 1, figsize=(12, 8 * len(experiments)))

for i, exp in enumerate(experiments):
    run_experiment(coords, *exp, ax=axes[i])

plt.tight_layout()
plt.savefig("zad2res.png")

# 715.8310488937599
# Optimal path: [(44, 17), (40, 22), (52, 15), (54, 26), (59, 31), (64, 23), (68, 14), (77, 8), (84, 14), (87, 10), (89, 12), (89, 4), (93, 4), (93, 5), (84, 22), (65, 40), (46, 49), (38, 54), (33, 64), (24, 63), (13, 54), (30, 38), (36, 38), (24, 33), (18, 21), (22, 9), (6, 1), (41, 76), (47, 78), (48, 82), (54, 78), (61, 69), (68, 72), (81, 74), (84, 67), (92, 76), (88, 85), (80, 91), (72, 90), (70, 87), (68, 97), (64, 98), (62, 97), (48, 100), (35, 99), (13, 90), (9, 93), (92, 56), (95, 60), (91, 50), (44, 17)]
# Ant count: 10, alpha: 0.5, beta: 1.2, pheromone evaporation rate: 0.4, pheromone constant: 1000.0, iterations: 30

# 671.8319921557573
# Optimal path: [(92, 76), (88, 85), (80, 91), (72, 90), (70, 87), (68, 97), (64, 98), (62, 97), (48, 100), (35, 99), (48, 82), (47, 78), (41, 76), (54, 78), (61, 69), (68, 72), (81, 74), (84, 67), (95, 60), (92, 56), (91, 50), (65, 40), (59, 31), (54, 26), (64, 23), (68, 14), (77, 8), (84, 14), (87, 10), (89, 12), (89, 4), (93, 4), (93, 5), (84, 22), (52, 15), (44, 17), (40, 22), (36, 38), (30, 38), (24, 33), (18, 21), (22, 9), (6, 1), (13, 54), (24, 63), (33, 64), (38, 54), (46, 49), (13, 90), (9, 93), (92, 76)]
# Ant count: 20, alpha: 0.7, beta: 1.0, pheromone evaporation rate: 0.3, pheromone constant: 500.0, iterations: 50

# 671.8319921557573
# Optimal path: [(92, 76), (88, 85), (80, 91), (72, 90), (70, 87), (68, 97), (64, 98), (62, 97), (48, 100), (35, 99), (48, 82), (47, 78), (41, 76), (54, 78), (61, 69), (68, 72), (81, 74), (84, 67), (95, 60), (92, 56), (91, 50), (65, 40), (59, 31), (54, 26), (64, 23), (68, 14), (77, 8), (84, 14), (87, 10), (89, 12), (89, 4), (93, 4), (93, 5), (84, 22), (52, 15), (44, 17), (40, 22), (36, 38), (30, 38), (24, 33), (18, 21), (22, 9), (6, 1), (13, 54), (24, 63), (33, 64), (38, 54), (46, 49), (13, 90), (9, 93), (92, 76)]
# Ant count: 60, alpha: 0.9, beta: 1.5, pheromone evaporation rate: 0.2, pheromone constant: 2000.0, iterations: 300

# 598.4859706027403
# Optimal path: [(68, 72), (61, 69), (54, 78), (47, 78), (48, 82), (41, 76), (33, 64), (24, 63), (13, 54), (30, 38), (36, 38), (24, 33), (18, 21), (22, 9), (6, 1), (40, 22), (44, 17), (52, 15), (54, 26), (59, 31), (64, 23), (68, 14), (77, 8), (84, 14), (87, 10), (89, 12), (89, 4), (93, 4), (93, 5), (84, 22), (65, 40), (46, 49), (38, 54), (13, 90), (9, 93), (35, 99), (48, 100), (62, 97), (64, 98), (68, 97), (72, 90), (70, 87), (80, 91), (88, 85), (92, 76), (81, 74), (84, 67), (95, 60), (92, 56), (91, 50), (68, 72)]
# Ant count: 100, alpha: 0.6, beta: 1.1, pheromone evaporation rate: 0.5, pheromone constant: 1500.0, iterations: 200