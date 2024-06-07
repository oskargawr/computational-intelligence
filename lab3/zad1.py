import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=285789)

# print(test_set)


def classify_iris(sl, sw, pl, pw):
    if pw < 1:
        return "Setosa"
    elif pw >= 1 and pw < 1.8:
        return "Versicolor"
    else:
        return "Virginica"


good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    sl = test_set.iloc[i, 0]
    sw = test_set.iloc[i, 1]
    pl = test_set.iloc[i, 2]
    pw = test_set.iloc[i, 3]
    predicted = classify_iris(sl, sw, pl, pw)
    actual = test_set.iloc[i, 4]
    if predicted == actual:
        good_predictions += 1

print("Accuracy: ", good_predictions / len)
print("Number of good predictions: ", good_predictions)

# print all values in test set sorted by variety
train_set = train_set.sort_values(by="variety")
pd.set_option("display.max_rows", None)
# print(train_set)
