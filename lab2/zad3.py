from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

iris = datasets.load_iris()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# wykres normalny
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("oryginalne dane: ", df.describe())
scatter = axes[0].scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
axes[0].set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = axes[0].legend(scatter.legend_elements()[0], iris.target_names)
axes[0].set_title("Normal")

# wykres po znormalizowaniu min-max
df = pd.DataFrame(iris.data, columns=iris.feature_names)

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("--------------------")
print("dane po normalizacji min-max:", df_scaled.describe())

scatter = axes[1].scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=iris.target)
axes[1].set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = axes[1].legend(scatter.legend_elements()[0], iris.target_names)
axes[1].set_title("Min-Max Normalization")

# wykres po znormalizowaniu z-score
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("--------------------")
print("dane po normalizacji z-score: ", df_scaled.describe())

scatter = axes[2].scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=iris.target)
axes[2].set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = axes[2].legend(scatter.legend_elements()[0], iris.target_names)
axes[2].set_title("Z-score Normalization")

plt.show()
