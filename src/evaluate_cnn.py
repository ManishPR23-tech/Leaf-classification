
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "cnn_best.keras"

CLASS_NAMES = ["hibiscus", "marigold", "rose"]
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

print("Loading saved CNN...")
model = keras.models.load_model(MODEL_PATH)

print("\nLoading test dataset...")
test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nEvaluating CNN on test data...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())

probabilities = model.predict(test_ds, verbose=1)
y_pred = probabilities.argmax(axis=1)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

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

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
