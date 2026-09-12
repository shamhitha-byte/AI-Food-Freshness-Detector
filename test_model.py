import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained model
model = tf.keras.models.load_model(
    "model/food_freshness_model.keras"
)

# Test dataset
test_path = "Dataset/archive/Quality Dataset/test"

# MobileNetV2 preprocessing
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

# Evaluate model
loss, accuracy = model.evaluate(test_data)

print("\n==============================")
print("TEST RESULTS")
print("==============================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
print("==============================")