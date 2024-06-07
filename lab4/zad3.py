import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from math import sqrt
from sklearn.metrics import r2_score

df = pd.read_csv("diabetes.csv")
print(df.shape)
print(df.describe().transpose())
# print(df.isnull().sum())

df["class"] = df["class"].replace(["tested_positive", "tested_negative"], [1, 0])
print(df.head())

target_column = ["class"]
predictors = list(set(list(df.columns)) - set(target_column))
df[predictors] = df[predictors] / df[predictors].max()
print(df.describe().transpose())


X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=40
)

# print(X_train.shape)
# print(X_test.shape)


mlp = MLPClassifier(
    hidden_layer_sizes=(6, 3), activation="relu", solver="adam", max_iter=500
)

mlp.fit(X_train, y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print("Train data")
print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))
print(accuracy_score(y_train, predict_train))

print("------")
print("Test data")
print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))
print(accuracy_score(y_test, predict_test))

# w tym przypadku gorsze jest false negative


# Train data
# [[320  38]
#  [ 81  98]]
#               precision    recall  f1-score   support

#            0       0.80      0.89      0.84       358
#            1       0.72      0.55      0.62       179

#     accuracy                           0.78       537
#    macro avg       0.76      0.72      0.73       537
# weighted avg       0.77      0.78      0.77       537

# 0.7783985102420856


# Test data
# [[125  17]
#  [ 40  49]]
#               precision    recall  f1-score   support

#            0       0.76      0.88      0.81       142
#            1       0.74      0.55      0.63        89

#     accuracy                           0.75       231
#    macro avg       0.75      0.72      0.72       231
# weighted avg       0.75      0.75      0.74       231

# 0.7532467532467533