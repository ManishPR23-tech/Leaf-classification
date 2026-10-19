
from pathlib import Path
import random
import shutil

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# Reproducible results: the same split each time
random.seed(42)

# Split proportions
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The three species we selected
CLASSES = ["hibiscus", "rose", "marigold"]


def main():
    for class_name in CLASSES:
        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Could not find the image folder: {class_dir}"
            )

        images = [
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            raise ValueError(f"No images found in {class_dir}")

        random.shuffle(images)

        train_end = int(len(images) * TRAIN_RATIO)
        val_end = train_end + int(len(images) * VAL_RATIO)

        split_groups = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in split_groups.items():
            destination = OUTPUT_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                destination_file = destination / image_path.name

                if not destination_file.exists():
                    shutil.copy2(image_path, destination_file)

            print(f"{class_name} - {split_name}: {len(split_images)} images")

    print("\nDataset splitting complete!")


if __name__ == "__main__":
    main()
