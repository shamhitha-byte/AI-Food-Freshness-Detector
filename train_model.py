import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ==========================================
# 1. Dataset paths
# ==========================================

train_path = "Dataset/archive/Quality Dataset/train"
valid_path = "Dataset/archive/Quality Dataset/valid"

# ==========================================
# 2. Settings
# ==========================================

IMG_SIZE = 160
BATCH_SIZE = 16

# ==========================================
# 3. Load training images
# ==========================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

valid_data = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print("\n================================")
print("CLASS MAPPING")
print("================================")
print(train_data.class_indices)

# ==========================================
# 4. Load MobileNetV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained layers
base_model.trainable = False

# ==========================================
# 5. Create our model
# ==========================================

model = models.Sequential([
    
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.4),

    layers.Dense(1, activation="sigmoid")
])

# ==========================================
# 6. Compile
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 7. Display model
# ==========================================

model.summary()

# ==========================================
# 8. Create model folder
# ==========================================

os.makedirs("model", exist_ok=True)

# ==========================================
# 9. Callbacks
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "model/best_food_freshness_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# ==========================================
# 10. Train the model
# ==========================================

print("\n================================")
print("STARTING TRAINING")
print("================================")

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=20,
    callbacks=[early_stopping, checkpoint]
)

# ==========================================
# 11. Save final model
# ==========================================

model.save("model/food_freshness_model.keras")

print("\n================================")
print("TRAINING COMPLETED")
print("================================")
print("Model saved successfully!")
print("Location: model/food_freshness_model.keras")
# ==========================================
# 12. Save Accuracy Graph
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig("model/accuracy_graph.png")
plt.close()


# ==========================================
# 13. Save Loss Graph
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig("model/loss_graph.png")
plt.close()

print("Accuracy graph saved!")
print("Loss graph saved!")