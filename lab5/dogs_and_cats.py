# %%
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

# %%
FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# %%
filenames = os.listdir("./dogs-cats-mini")

# %%
categories = []

for filename in filenames:
    category = filename.split(".")[0]
    if category == "dog":
        categories.append(1)
    else:
        categories.append(0)

# %%
df = pd.DataFrame({"filename": filenames, "category": categories})

# %%
df.head()

# %%
df.tail()

# %%
df["category"].value_counts().plot.bar(color=["lightblue", "coral"])

# %%
sample = random.choice(filenames)
image = load_img("./dogs-cats-mini/" + sample)
plt.imshow(image)

# %%
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Activation,
    BatchNormalization,
)

# %%
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

model.summary()

# %%
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# %%
earlystop = EarlyStopping(patience=10)

# %%
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, verbose=1, factor=0.5, min_lr=0.00001
)

# %%
callbacks = [earlystop, learning_rate_reduction]

# %%
df["category"] = df["category"].replace({0: "cat", 1: "dog"})

# %%
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# %%
train_df["category"].value_counts().plot.bar(color=["lightblue", "coral"])

# %%
validate_df["category"].value_counts().plot.bar(color=["lightblue", "coral"])

# %%
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

# %%
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "./dogs-cats-mini/",
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=batch_size,
)

# %%
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "./dogs-cats-mini/",
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=batch_size,
)

# %%
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df,
    "./dogs-cats-mini/",
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
)

# %%
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i + 1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

# %%
epochs = 3 if FAST_RUN else 5
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size,
    steps_per_epoch=total_train // batch_size,
    callbacks=callbacks,
)

# Save model
model.save("model.h5")

# Save training history
import pickle

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# %%
# Load model
from keras.models import load_model

model = load_model("model.h5")

# Load training history
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# %%
# Plot learning curves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history["loss"], color="b", label="Training loss")
ax1.plot(history["val_loss"], color="r", label="validation loss")
ax1.set_xticks(np.arange(1, epochs + 1, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history["accuracy"], color="b", label="Training accuracy")
ax2.plot(history["val_accuracy"], color="r", label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs + 1, 1))

legend = plt.legend(loc="best", shadow=True)
plt.tight_layout()
plt.show()

# %%
# Predict on test data
test_filenames = os.listdir("./test1")
test_df = pd.DataFrame({"filename": test_filenames})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "./test1",
    x_col="filename",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False,
)

predict = model.predict(test_generator, steps=int(np.ceil(nb_samples / batch_size)))

# %%
# Map predictions to categories
label_map = dict((v, k) for k, v in train_generator.class_indices.items())
test_df["category"] = [label_map[k] for k in np.argmax(predict, axis=1)]

# Replace categories with integers for plotting
test_df["category"] = test_df["category"].replace({"dog": 1, "cat": 0})

# Plot category distribution
test_df["category"].value_counts().plot.bar()

# %%
# Display sample test results
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row["filename"]
    category = row["category"]
    img = load_img("./test1/" + filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.xlabel(filename + "(" + "{}".format(category) + ")")
plt.tight_layout()
plt.show()

# %%
# Save submission file
submission_df = test_df.copy()
submission_df["id"] = submission_df["filename"].str.split(".").str[0]
submission_df["label"] = submission_df["category"]
submission_df.drop(["filename", "category"], axis=1, inplace=True)
submission_df.to_csv("submission.csv", index=False)
