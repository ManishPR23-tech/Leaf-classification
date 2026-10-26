
from pathlib import Path
import random
import shutil
import hashlib

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
UNKNOWN_DIR = PROJECT_DIR / "data" / "unknown_raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# Reproducible results
RANDOM_SEED = 42

# Split proportions
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KNOWN_CLASSES = [
    "hibiscus",
    "rose",
    "marigold",
    "lotus",
    "rose_periwinkle",
]

UNKNOWN_CLASSES = ["togor"]
SPLITS = ["train", "val", "test"]


def get_images(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def calculate_hash(path):
    """Hash exact file contents to detect identical files."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def remove_duplicate_contents(images, class_name):
    """Keep one copy of each exact image within a class."""
    unique_images = []
    seen_hashes = set()
    duplicates_removed = 0

    for image in images:
        digest = calculate_hash(image)

        if digest in seen_hashes:
            duplicates_removed += 1
            print(f"Duplicate excluded from {class_name}: {image.name}")
            continue

        seen_hashes.add(digest)
        unique_images.append(image)

    if duplicates_removed:
        print(
            f"{class_name}: excluded {duplicates_removed} "
            "duplicate source file(s)"
        )

    return unique_images


def main():
    # Validate every source folder before changing processed data.
    sources = {}

    for class_name in KNOWN_CLASSES:
        folder = RAW_DIR / class_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")

        images = get_images(folder)

        if len(images) != 400:
            raise ValueError(
                f"Expected 400 images in {folder}, found {len(images)}"
            )

        sources[class_name] = images

    for class_name in UNKNOWN_CLASSES:
        folder = UNKNOWN_DIR / class_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing unknown folder: {folder}")

        images = get_images(folder)

        if len(images) != 400:
            raise ValueError(
                f"Expected 400 images in {folder}, found {len(images)}"
            )

        sources["unknown"] = images

    # Exclude exact duplicates within each class.
    for class_name in sources:
        sources[class_name] = remove_duplicate_contents(
            sources[class_name], class_name
        )

    # Split the unique source images reproducibly.
    split_groups = {}

    for class_name, images in sources.items():
        shuffled = images.copy()
        random.Random(RANDOM_SEED).shuffle(shuffled)

        train_end = int(len(shuffled) * TRAIN_RATIO)
        val_end = train_end + int(len(shuffled) * VAL_RATIO)

        split_groups[class_name] = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

    # Rebuild only the processed split directories.
    # Original images in data/raw and data/unknown_raw are untouched.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in SPLITS:
        split_dir = OUTPUT_DIR / split_name

        if split_dir.exists():
            shutil.rmtree(split_dir)

        split_dir.mkdir(parents=True)

    # Copy each selected image into its split.
    for class_name, groups in split_groups.items():
        for split_name, images in groups.items():
            destination = OUTPUT_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for image in images:
                shutil.copy2(image, destination / image.name)

            print(f"{class_name} - {split_name}: {len(images)} images")

    total_images = sum(len(images) for images in sources.values())

    print("\nDataset splitting complete!")
    print("Unique source images used:", total_images)
    print("Original source files were not deleted.")


if __name__ == "__main__":
    main()
