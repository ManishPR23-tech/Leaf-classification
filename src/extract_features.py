
from pathlib import Path
import numpy as np
from utils.preprocessing import preprocess_image

# Find the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

# All six classes the model must learn
CLASSES = [
    "hibiscus",
    "marigold",
    "rose",
    "lotus",
    "rose_periwinkle",
    "unknown",
]

# Dataset splits
SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def load_dataset(split):
    """Load images and labels for one dataset split."""
    if split not in SPLITS:
        raise ValueError(
            f"Invalid split: {split}. Choose from {SPLITS}"
        )

    images = []
    labels = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / split / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            raise ValueError(
                f"No images found in {class_dir}"
            )

        for image_path in image_paths:
            image = preprocess_image(str(image_path))
            features = image.reshape(-1)

            images.append(features)
            labels.append(class_name)

        print(f"{split}/{class_name}: {len(image_paths)} images")

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
