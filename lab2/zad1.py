import pandas as pd
import numpy as np

df = pd.read_csv("iris_with_errors.csv")

missings_values = ["n/a", "na", "--", "-", "nan", "NaN", "N/A", "NA", "NAN"]
df = pd.read_csv("iris_with_errors.csv", na_values=missings_values)

print("Rows with NaN values:")
print(df[df.isnull().any(axis=1)])
print("--------------------")
print(df.isnull().sum())
print("--------------------")


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


for col in df.select_dtypes(include=[np.number]).columns:
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median()
        df[col] = df[col].apply(lambda x: median if x <= 0 or x > 15 else x)

df["variety"] = df["variety"].str.capitalize()

epsilon = 2
for row in df.index:
    if df.loc[row, "variety"] not in ["Setosa", "Versicolor", "Virginica"]:
        print(f"Row {row} has invalid species: {df.loc[row, 'variety']}")
        distances = {
            species: levenshtein_distance(df.loc[row, "variety"], species)
            for species in ["Setosa", "Versicolor", "Virginica"]
        }
        min_distance = min(distances.values())

        if min_distance <= epsilon:
            df.loc[row, "variety"] = min(distances, key=distances.get)
            print(f"Fixed to: {df.loc[row, 'variety']}")
        else:
            print("Species not recognized")
            df.drop(row, inplace=True)
