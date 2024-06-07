import matplotlib.pyplot as plt
import random
from aco import AntColony

plt.style.use("dark_background")


# Generowanie losowych współrzędnych
def generate_random_coords(n):
    return [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n)]


# Funkcje pomocnicze do rysowania
def plot_nodes(coords, w=12, h=8):
    for x, y in coords:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_optimal_path(coords, optimal_nodes):
    for i in range(len(optimal_nodes) - 1):
        plt.plot(
            (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
            (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
        )
    plt.plot(
        (optimal_nodes[-1][0], optimal_nodes[0][0]),
        (optimal_nodes[-1][1], optimal_nodes[0][1]),
    )
    plt.show()


# Eksperymenty z różnymi zestawami danych i parametrami
def run_experiment(
    n,
    ant_count,
    alpha,
    beta,
    pheromone_evaporation_rate,
    pheromone_constant,
    iterations,
):
    coords = generate_random_coords(n)
    plot_nodes(coords)

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
    plot_optimal_path(coords, optimal_nodes)

    print(f"Number of vertices: {n}")
    print(f"Optimal path: {optimal_nodes}")
    print(
        f"Ant count: {ant_count}, alpha: {alpha}, beta: {beta}, pheromone evaporation rate: {pheromone_evaporation_rate}, pheromone constant: {pheromone_constant}, iterations: {iterations}"
    )
    print("\n")


experiments = [
    (10, 300, 0.5, 1.2, 0.4, 1000.0, 300),
    (15, 300, 0.7, 1.0, 0.3, 500.0, 500),
    (20, 500, 0.9, 1.5, 0.2, 2000.0, 400),
    (25, 400, 0.6, 1.1, 0.5, 1500.0, 600),
]

# liczba wierzcholkow, liczba mrowek, parametr alfa (im wyzszy tym wieksze znaczenie feromonow), parametr beta (im wyzszy tym wieksze znaczenie odleglosci), wspolczynnik parowania feromonow, stala feromonowa, liczba iteracji
for exp in experiments:
    run_experiment(*exp)
