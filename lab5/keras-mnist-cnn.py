import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History, ModelCheckpoint

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
#   - Reshape: zmienia kasztalt danych wejsciowych, tak zeby pasowaly do wejsc modelu CNN.
#   W tym przypadku zmieniamy dane z ksztaltu (60000, 28, 28) na (60000, 28, 28, 1), aby dodać wymiar kanału.
#   - to_categorical: zmienia etykiety klasowe na macierz wskaźników binarnych (one-hot encoding).
#   jest to wymagane dla sieci neuronowej przy użyciu cross validation jako funkcji straty.
#   - np.argmax: znajduje indeks najwiekszej wartości wzdłuż osi. używamy tego, aby przywrócić oryginalne etykiety testowe dla confusion matrix.
train_images = (
    train_images.reshape((train_images.shape[0], 28, 28, 1)).astype("float32") / 255
)
test_images = (
    test_images.reshape((test_images.shape[0], 28, 28, 1)).astype("float32") / 255
)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(
    test_labels, axis=1
)  # Save original labels for confusion matrix

# Define model
model = Sequential(
    [
        Conv2D(
            32, (3, 3), activation="relu", input_shape=(28, 28, 1)
        ),  # 1. warstwa: konwolucyjna, 32 filtry, okno 3x3, ReLU
        MaxPooling2D((2, 2)),  # 2. warstwa: max pooling, okno 2x2
        Flatten(),  # 3. warstwa: spłaszczenie danych do jednowymiarowego wektora
        Dense(64, activation="relu"),  # 4. warstwa: pełni połączona, 64 neurony, ReLU
        Dense(
            10, activation="softmax"
        ),  # 5. warstwa: pełni połączona, 10 neuronów (jednostek wyjściowych), softmax (klasyfikacja)
    ]
)

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
history = History()
# e) Jak zmodyfikować kod programu, aby model sieci był zapisywany do pliku h5 co epokę, pod warunkiem, że w tej epoce osiągnęliśmy lepszy wynik?
# Dodajemy callback ModelCheckpoint, który zapisuje model tylko wtedy, gdy poprawiła się jego dokładność walidacji.
checkpoint = ModelCheckpoint(
    "best_model.keras",  # Change file extension to .keras
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
)

model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[history, checkpoint],
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# c) Jakich błędów na macierzy błędów jest najwięcej. Które cyfry są często mylone z jakimi innymi?
# najczesciej mylilo 8 z 2 i 4 z 9

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", color="grey")
plt.legend()
plt.tight_layout()
plt.show()

# d) Co możesz powiedzieć o krzywych uczenia się. Czy mamy przypadek przeuczenia lub niedouczenia się?
# krzywe uczenia sa zbiezne, nie ma przeuczenia ani niedouczenia

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()
