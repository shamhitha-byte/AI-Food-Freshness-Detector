import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model(
    "model/food_freshness_model.keras"
)

test_path = "Dataset/archive/Quality Dataset/test"

for class_name in ["fresh", "rotten"]:

    folder = os.path.join(test_path, class_name)

    print("\nActual:", class_name.upper())

    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)

        try:
            img = image.load_img(
                filepath,
                target_size=(128, 128)
            )

            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(
                img_array,
                verbose=0
            )[0][0]

            if prediction >= 0.5:
                predicted_class = "rotten"
                confidence = prediction * 100
            else:
                predicted_class = "fresh"
                confidence = (1 - prediction) * 100

            result = "✓" if predicted_class == class_name else "✗"

            print(
                result,
                filename,
                "→",
                predicted_class,
                f"({confidence:.1f}%)"
            )

        except Exception as e:
            print("Error:", filename, e)