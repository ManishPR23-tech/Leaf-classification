
from pathlib import Path
import numpy as np
from utils.preprocessing import preprocess_image

# Find the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

# Classes our model will learn to recognize
CLASSES = ["hibiscus", "marigold", "rose"]

# Dataset splits
SPLITS = ["train", "val", "test"]


def load_dataset(split):
    """Load images and labels for one dataset split."""
    images = []
    labels = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / split / class_name

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                continue

            image = preprocess_image(str(image_path))

            # Flatten the image into a one-dimensional feature vector
            features = image.reshape(-1)

            images.append(features)
            labels.append(class_name)

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels)

    return X, y


if __name__ == "__main__":
    for split in SPLITS:
        X, y = load_dataset(split)

        print(f"\n{split.upper()} DATASET")
        print("Feature matrix shape:", X.shape)
        print("Labels shape:", y.shape)
        print("Classes:", np.unique(y))
        print("Data type:", X.dtype)
