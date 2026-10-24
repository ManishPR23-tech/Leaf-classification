
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
TEST_DIR = PROJECT_DIR / "data" / "processed_clean" / "test"
MODEL_PATH = PROJECT_DIR / "models" / "mobilenet_clean_best.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
CLASS_NAMES = ["hibiscus", "marigold", "rose"]

# Load the clean model
print("Loading clean MobileNetV2 model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load the clean test dataset
print("\nLoading clean test dataset...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Collect actual labels
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

# Predict class probabilities
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

print(f"\nClean test accuracy: {accuracy:.4%}")
print(f"Correct predictions: {(y_true == y_pred).sum()} / {len(y_true)}")
print(f"Incorrect predictions: {(y_true != y_pred).sum()} / {len(y_true)}")

# Detailed classification metrics
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

# Confusion matrix: rows are actual classes,
# columns are predicted classes
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
