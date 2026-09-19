import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 160

MODEL_PATH = "model/best_food_freshness_model_v2.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("\n==============================")
print("MODEL CHECK")
print("==============================")

print("Model loaded successfully.")

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "Dataset_v2/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    label_mode="binary",
    shuffle=False
)

print("Class names:", test_ds.class_names)

# Check first 5 images
print("\nChecking first 5 test images...\n")

for images, labels in test_ds.take(5):

    image = images[0].numpy()
    actual_label = int(labels[0].numpy())

    # Same preprocessing used in app.py
    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(
        image,
        verbose=0
    )[0][0]

    if prediction >= 0.5:
        predicted = "ROTTEN"
    else:
        predicted = "FRESH"

    actual = test_ds.class_names[actual_label]

    print("--------------------------------")
    print("Actual    :", actual.upper())
    print("Prediction:", predicted)
    print("Raw value :", float(prediction))