import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree

df = pd.read_csv("iris.csv")

print(df.dtypes)

print(df.describe())

df["petal.width"].plot.hist()
plt.show()

sns.pairplot(df, hue="variety")
plt.show()

all_inputs = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
all_classes = df["variety"].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(
    all_inputs, all_classes, train_size=0.7, random_state=13
)

dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
print(dtc.score(test_inputs, test_classes))

plt.figure(figsize=(15, 10))
tree.plot_tree(dtc, filled=True)
plt.show()

cm = confusion_matrix(test_classes, dtc.predict(test_inputs))
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
