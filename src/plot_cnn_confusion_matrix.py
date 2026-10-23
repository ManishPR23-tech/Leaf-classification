
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "cnn_best.keras"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["hibiscus", "marigold", "rose"]
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

print("Loading CNN...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading test dataset...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())

probabilities = model.predict(test_ds, verbose=1)
y_pred = probabilities.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)

plt.title("CNN Confusion Matrix - Test Set")
plt.xlabel("Predicted Flower")
plt.ylabel("Actual Flower")
plt.tight_layout()

output_path = FIGURE_DIR / "cnn_confusion_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Confusion matrix saved to: {output_path}")
print("\nConfusion matrix:")
print(cm)
