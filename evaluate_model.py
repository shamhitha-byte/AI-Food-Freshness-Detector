import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# ==========================================
# 1. Load trained model
# ==========================================

model = tf.keras.models.load_model(
    "model/food_freshness_model.keras"
)

print("Model loaded successfully!")


# ==========================================
# 2. Test dataset
# ==========================================

test_path = "Dataset/archive/Quality Dataset/test"

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(160, 160),
    batch_size=16,
    class_mode="binary",
    shuffle=False
)


# ==========================================
# 3. Evaluate model
# ==========================================

loss, accuracy = model.evaluate(test_data)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")


# ==========================================
# 4. Predictions
# ==========================================

predictions = model.predict(test_data)

predicted_classes = (predictions >= 0.5).astype(int).flatten()

actual_classes = test_data.classes


# ==========================================
# 5. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


# Display confusion matrix

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fresh", "Rotten"]
)

display.plot()

plt.title("Food Freshness Confusion Matrix")

plt.savefig(
    "model/confusion_matrix.png"
)

plt.show()


# ==========================================
# 6. Classification Report
# ==========================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        actual_classes,
        predicted_classes,
        target_names=["Fresh", "Rotten"]
    )
)


print("\n==============================")
print("EVALUATION COMPLETED")
print("==============================")