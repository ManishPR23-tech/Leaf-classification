
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "mobilenet_6class_best.keras"

CLASS_NAMES = [
    "hibiscus",
    "lotus",
    "marigold",
    "rose",
    "rose_periwinkle",
    "unknown",
]

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

def main():
    print("Loading six-class MobileNetV2 model...")
    model = keras.models.load_model(MODEL_PATH)

    print("\nLoading test dataset...")
    test_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\nEvaluating on test data...")
    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("\nConfusion matrix")
    print("Rows = actual classes; columns = predicted classes")
    print(CLASS_NAMES)
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    ))

if __name__ == "__main__":
    main()
