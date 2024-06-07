import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
import pandas as pd
from sklearn import datasets, decomposition

iris = datasets.load_iris()
X = iris.data
Y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()

ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.set_position([0, 0, 0.95, 1])

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [("Setosa", 0), ("Versicolor", 1), ("Virginica", 2)]:
    ax.text3D(
        X[Y == label, 0].mean(),
        X[Y == label, 1].mean(),
        X[Y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )

Y = np.choose(Y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# plt.show()

# 2)

df = pd.DataFrame(iris.data, columns=iris.feature_names)

total_variance = df.var().sum()
# print(f"Total variance: {total_variance}")

sorted_columns = df.var().sort_values(ascending=True).index.tolist()
print(f"Columns sorted by variance: {sorted_columns}")

df_copy = df.copy()
removed_columns = []
for column in sorted_columns:
    df_copy = df_copy.drop(columns=[column])
    remaining_variance = df_copy.var().sum()
    # print(remaining_variance / total_variance)
    if remaining_variance / total_variance < 0.95:
        break
    removed_columns.append(column)

print(f"Columns removed: {removed_columns}")

df_new = df.drop(columns=removed_columns)

# PCA again
pca_new = decomposition.PCA(n_components=3)
pca_new.fit(df_new)
X_new = pca_new.transform(df_new)

# Plotting the new PCA
fig_new = plt.figure(2, figsize=(4, 3))
plt.clf()

ax_new = fig_new.add_subplot(111, projection="3d", elev=48, azim=134)
ax_new.set_position([0, 0, 0.95, 1])

plt.cla()

for name, label in [("Setosa", 0), ("Versicolor", 1), ("Virginica", 2)]:
    ax_new.text3D(
        X_new[Y == label, 0].mean(),
        X_new[Y == label, 1].mean(),
        X_new[Y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )

ax_new.scatter(
    X_new[:, 0], X_new[:, 1], X_new[:, 2], c=Y, cmap=plt.cm.nipy_spectral, edgecolor="k"
)

ax_new.xaxis.set_ticklabels([])
ax_new.yaxis.set_ticklabels([])
ax_new.zaxis.set_ticklabels([])

plt.show()
