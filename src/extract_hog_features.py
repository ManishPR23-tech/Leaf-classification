
from pathlib import Path

import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray

from utils.preprocessing import preprocess_image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

CLASSES = ["hibiscus", "marigold", "rose"]
SPLITS = ["train", "val", "test"]


def extract_hog(image):
    """Convert a preprocessed RGB image into HOG features."""
    gray = rgb2gray(image)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return features.astype(np.float32)


def load_hog_dataset(split):
    """Load HOG features and labels for one dataset split."""
    features_list = []
    labels = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / split / class_name

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                continue

            image = preprocess_image(str(image_path))
            features = extract_hog(image)

            features_list.append(features)
            labels.append(class_name)

    X = np.asarray(features_list, dtype=np.float32)
    y = np.asarray(labels)

    return X, y


if __name__ == "__main__":
    for split in SPLITS:
        print(f"\nLoading {split} dataset...")
        X, y = load_hog_dataset(split)

        print("Feature matrix shape:", X.shape)
        print("Labels shape:", y.shape)
        print("Classes:", np.unique(y))
        print("Data type:", X.dtype)
