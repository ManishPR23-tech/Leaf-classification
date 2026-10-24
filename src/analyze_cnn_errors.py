
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "test"
MODEL_PATH = PROJECT_DIR / "models" / "cnn_best.keras"
OUTPUT_PATH = PROJECT_DIR / "reports" / "figures" / "cnn_misclassified_images.png"

CLASS_NAMES = ["hibiscus", "marigold", "rose"]
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

print("Loading saved CNN...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading test images...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Keep the images and their true labels in the same order.
images_list = []
labels_list = []

for images, labels in test_ds:
    images_list.append(images.numpy())
    labels_list.extend(labels.numpy())

images = np.concatenate(images_list, axis=0)
y_true = np.array(labels_list)

print("Predicting test images...")
probabilities = model.predict(test_ds, verbose=1)
y_pred = probabilities.argmax(axis=1)

wrong_indices = np.where(y_true != y_pred)[0]

print(f"\nMisclassified images: {len(wrong_indices)}")
print("Showing up to the first 12 errors.")

# Display up to 12 errors.
selected = wrong_indices[:12]

if len(selected) == 0:
    print("No misclassified images found.")
else:
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, selected):
        ax.imshow(images[idx].astype("uint8"))
        ax.set_title(
            f"Actual: {CLASS_NAMES[y_true[idx]]}\n"
            f"Predicted: {CLASS_NAMES[y_pred[idx]]}\n"
            f"Confidence: {probabilities[idx, y_pred[idx]]:.1%}"
        )
        ax.axis("off")

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Error visualization saved to: {OUTPUT_PATH}")

print("\nErrors by actual class:")
for class_id, class_name in enumerate(CLASS_NAMES):
    count = int(np.sum((y_true == class_id) & (y_pred != class_id)))
    print(f"{class_name}: {count}")
