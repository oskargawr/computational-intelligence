from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

iris = load_iris()
datasets = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

train_data, test_data, train_labels, test_labels = datasets

scaler = StandardScaler()
scaler.fit(train_data)


train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

print(train_data[:3])

mlp = MLPClassifier(hidden_layer_sizes=(4, 2, 1), max_iter=1000)

mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
predictions_test = mlp.predict(test_data)

# print(confusion_matrix(train_labels, predictions_train))
# print(classification_report(train_labels, predictions_train))
# print(accuracy_score(train_labels, predictions_train))

# print(confusion_matrix(test_labels, predictions_test))
# print(classification_report(test_labels, predictions_test))
print(accuracy_score(test_labels, predictions_test))
# 0.28888888888888886

mlp2 = MLPClassifier(hidden_layer_sizes=(4, 3, 1), max_iter=1000)

mlp2.fit(train_data, train_labels)

predictions_train2 = mlp2.predict(train_data)
predictions_test2 = mlp2.predict(test_data)

# print(confusion_matrix(train_labels, predictions_train2))
# print(classification_report(train_labels, predictions_train2))
# print(accuracy_score(train_labels, predictions_train2))

# print(confusion_matrix(test_labels, predictions_test2))
# print(classification_report(test_labels, predictions_test2))
print(accuracy_score(test_labels, predictions_test2))
# 0.5555555555555556

mlp3 = MLPClassifier(hidden_layer_sizes=(4, 3, 3, 1), max_iter=1000)

mlp3.fit(train_data, train_labels)

predictions_train3 = mlp3.predict(train_data)
predictions_test3 = mlp3.predict(test_data)

# print(confusion_matrix(train_labels, predictions_train3))
# print(classification_report(train_labels, predictions_train3))
# print(accuracy_score(train_labels, predictions_train3))

# print(confusion_matrix(test_labels, predictions_test3))
# print(classification_report(test_labels, predictions_test3))
print(accuracy_score(test_labels, predictions_test3))
# 0.9777777777777777

mlp4 = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)

mlp4.fit(train_data, train_labels)

predictions_train4 = mlp4.predict(train_data)
predictions_test4 = mlp4.predict(test_data)

# print(confusion_matrix(train_labels, predictions_train4))
# print(classification_report(train_labels, predictions_train4))
# print(accuracy_score(train_labels, predictions_train4))

# print(confusion_matrix(test_labels, predictions_test4))
# print(classification_report(test_labels, predictions_test4))
print(accuracy_score(test_labels, predictions_test4))
# 1.0
