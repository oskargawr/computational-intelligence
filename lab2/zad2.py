import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
import pandas as pd
from sklearn import datasets, decomposition

iris = datasets.load_iris()

pca_iris = decomposition.PCA().fit(iris.data)

explained_variance_ratio = pca_iris.explained_variance_ratio_

cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
print(f"Explained variance ratio: {explained_variance_ratio}")
print(f"Cumulative variance ratio: {cumulative_variance_ratio}")

n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

print(f"Number of components to explain 95% of variance: {n_components}")

X = iris.data
Y = iris.target

ipca = decomposition.IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)

pca = decomposition.PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

colors = ["navy", "turquoise", "darkorange"]

for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(
            X_transformed[Y == i, 0],
            X_transformed[Y == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )

    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(
            title + " of iris dataset\nMean absolute unsigned error " "%.6f" % err
        )
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

plt.show()
