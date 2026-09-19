import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 160
BATCH_SIZE = 32

TEST_DIR = "Dataset_v2/test"
MODEL_PATH = "model/best_food_freshness_model_v2.keras"

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class names:", test_ds.class_names)

# Load trained V2 model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_ds)

print("\n==============================")
print("V2 TEST RESULTS")
print("==============================")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
predictions = model.predict(test_ds)
predicted_classes = (predictions >= 0.5).astype(int).flatten()

# Actual labels
actual_classes = np.concatenate([
    y.numpy().flatten()
    for x, y in test_ds
])

# Confusion matrix
cm = confusion_matrix(actual_classes, predicted_classes)

print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(
    actual_classes,
    predicted_classes,
    target_names=test_ds.class_names
))

# Save confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("V2 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], test_ds.class_names)
plt.yticks([0, 1], test_ds.class_names)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.savefig("model/confusion_matrix_v2.png")
plt.close()

print("\nConfusion matrix saved!")
print("\nV2 testing completed!")