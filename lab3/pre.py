import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=13)

print(test_set)
# print(test_set.shape[0])

train_inputs = train_set.iloc[:, 0:4]
train_classes = train_set.iloc[:, 4]
test_inputs = test_set.iloc[:, 0:4]
test_classes = test_set.iloc[:, 4]

print("Train inputs: ", train_inputs)
print("Train classes: ", train_classes)
print("Test inputs: ", test_inputs)
print("Test classes: ", test_classes)
