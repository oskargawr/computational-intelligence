import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

sns.set()


df = pd.read_csv("iris.csv")

X = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
y = df["variety"]
y = pd.get_dummies(y)

print("X: ", X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=13
)

knn3 = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn3.fit(X_train, y_train)

y_pred3 = knn3.predict(X_test)

cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred3.argmax(axis=1))

print("Accuracy for 3 neighbors: ", knn3.score(X_test, y_test))
print("Confusion matrix for 3 neighbors: ", cm)

knn5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn5.fit(X_train, y_train)

y_pred5 = knn5.predict(X_test)

print("Accuracy for 5 neighbors: ", knn5.score(X_test, y_test))
print(
    "Confusion matrix for 5 neighbors: ",
    confusion_matrix(y_test.values.argmax(axis=1), y_pred5.argmax(axis=1)),
)

knn11 = KNeighborsClassifier(n_neighbors=11, metric="euclidean")
knn11.fit(X_train, y_train)

y_pred11 = knn11.predict(X_test)

print("Accuracy for 11 neighbors: ", knn11.score(X_test, y_test))
print(
    "Confusion matrix for 11 neighbors: ",
    confusion_matrix(y_test.values.argmax(axis=1), y_pred11.argmax(axis=1)),
)


### Naive Bayes
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(
    "Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0], (y_test != y_pred).sum())
)
print("Accuracy: ", gnb.score(X_test, y_test))
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
