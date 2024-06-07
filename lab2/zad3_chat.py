import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Wczytanie danych
iris = datasets.load_iris()
X = iris.data[:, :2]  # tylko dwie zmienne: sepal length i sepal width

# Dane oryginalne
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Dane oryginalne")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

# Normalizacja min-max
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
plt.subplot(1, 3, 2)
plt.scatter(X_minmax[:, 0], X_minmax[:, 1])
plt.title("Normalizacja Min-Max")
plt.xlabel("Sepal Length (min-max)")
plt.ylabel("Sepal Width (min-max)")

# Normalizacja z-score
scaler_zscore = StandardScaler()
X_zscore = scaler_zscore.fit_transform(X)
plt.subplot(1, 3, 3)
plt.scatter(X_zscore[:, 0], X_zscore[:, 1])
plt.title("Normalizacja Z-Score")
plt.xlabel("Sepal Length (z-score)")
plt.ylabel("Sepal Width (z-score)")

plt.tight_layout()
plt.show()

# Statystyki dla danych oryginalnych
print("Statystyki dla danych oryginalnych:")
print(
    "Sepal Length: Min =",
    np.min(X[:, 0]),
    ", Max =",
    np.max(X[:, 0]),
    ", Średnia =",
    np.mean(X[:, 0]),
    ", Odchylenie standardowe =",
    np.std(X[:, 0]),
)
print(
    "Sepal Width: Min =",
    np.min(X[:, 1]),
    ", Max =",
    np.max(X[:, 1]),
    ", Średnia =",
    np.mean(X[:, 1]),
    ", Odchylenie standardowe =",
    np.std(X[:, 1]),
)

# Statystyki dla danych po normalizacji Min-Max
print("\nStatystyki dla danych po normalizacji Min-Max:")
print(
    "Sepal Length: Min =",
    np.min(X_minmax[:, 0]),
    ", Max =",
    np.max(X_minmax[:, 0]),
    ", Średnia =",
    np.mean(X_minmax[:, 0]),
    ", Odchylenie standardowe =",
    np.std(X_minmax[:, 0]),
)
print(
    "Sepal Width: Min =",
    np.min(X_minmax[:, 1]),
    ", Max =",
    np.max(X_minmax[:, 1]),
    ", Średnia =",
    np.mean(X_minmax[:, 1]),
    ", Odchylenie standardowe =",
    np.std(X_minmax[:, 1]),
)

# Statystyki dla danych po normalizacji Z-Score
print("\nStatystyki dla danych po normalizacji Z-Score:")
print(
    "Sepal Length: Min =",
    np.min(X_zscore[:, 0]),
    ", Max =",
    np.max(X_zscore[:, 0]),
    ", Średnia =",
    np.mean(X_zscore[:, 0]),
    ", Odchylenie standardowe =",
    np.std(X_zscore[:, 0]),
)
print(
    "Sepal Width: Min =",
    np.min(X_zscore[:, 1]),
    ", Max =",
    np.max(X_zscore[:, 1]),
    ", Średnia =",
    np.mean(X_zscore[:, 1]),
    ", Odchylenie standardowe =",
    np.std(X_zscore[:, 1]),
)

# prompt:
# Na podstawie datasetow irysów:
# iris = datasets.load_iris()
# dla dwóch zmiennych sepal length i sepal width, wykonaj trzy wykresy. Pierwszy wykres będzie przedstawiał dane oryginalne,
# drugi wykres będzie przedstawiał dane po normalizacji min-max, a trzeci dane po normalizacji z-score. Co możesz powiedzieć
# o min, max, mean, standard deviation dla tych danych?
