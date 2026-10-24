
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_DIR / "data" / "processed" / "test"
MODEL_PATH = PROJECT_DIR / "models" / "mobilenet_best.keras"

# Settings
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
CLASS_NAMES = ["hibiscus", "marigold", "rose"]

# Load the saved model
print("Loading saved MobileNetV2 model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load test dataset
print("\nLoading test dataset...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Get actual labels and predictions
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds)
y_pred = np.argmax(probabilities, axis=1)

# Overall evaluation
accuracy = accuracy_score(y_true, y_pred)

print(f"\nTest accuracy: {accuracy:.4%}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))

# Inspect the test image that duplicates a training image
duplicate_name = "download (9).jpeg"
duplicate_path = TEST_DIR / "rose" / duplicate_name

print("\n--- Duplicate Image Check ---")

if duplicate_path.exists():
    image = tf.keras.utils.load_img(
        duplicate_path,
        target_size=IMAGE_SIZE,
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    image_probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(image_probabilities))

    print("Image:", duplicate_name)
    print("Actual class: rose")
    print("Predicted class:", CLASS_NAMES[predicted_index])
    print(
        "Prediction confidence:",
        f"{image_probabilities[predicted_index] * 100:.2f}%",
    )
else:
    print("Duplicate image was not found in the test folder.")
