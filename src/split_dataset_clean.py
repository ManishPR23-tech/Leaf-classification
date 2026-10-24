
from pathlib import Path
from collections import defaultdict
import hashlib
import random
import shutil

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed_clean"

CLASS_NAMES = ["hibiscus", "marigold", "rose"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)


def get_hash(path):
    """Return a hash of the file's contents."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    # Group identical images together within each class.
    class_groups = {}

    for class_name in CLASS_NAMES:
        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        groups = defaultdict(list)

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                groups[get_hash(path)].append(path)

        class_groups[class_name] = list(groups.values())

    # Prepare a new output directory without touching existing data.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Rename or move it before running this script again."
        )

    OUTPUT_DIR.mkdir(parents=True)

    total_images = 0

    for class_name, groups in class_groups.items():
        random.shuffle(groups)

        total_groups = len(groups)
        train_end = int(total_groups * TRAIN_RATIO)
        val_end = train_end + int(total_groups * VAL_RATIO)

        split_groups = {
            "train": groups[:train_end],
            "val": groups[train_end:val_end],
            "test": groups[val_end:],
        }

        print(f"\nClass: {class_name}")

        for split_name, selected_groups in split_groups.items():
            destination = OUTPUT_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            count = 0

            for group in selected_groups:
                # Keep identical files together in the same split.
                for source in group:
                    target = destination / source.name

                    # Avoid overwriting files with the same filename.
                    if target.exists():
                        target = destination / (
                            f"{source.stem}_{get_hash(source)[:8]}"
                            f"{source.suffix}"
                        )

                    shutil.copy2(source, target)
                    count += 1

            total_images += count
            print(f"  {split_name}: {count} images")

    print("\nClean dataset created successfully.")
    print("Location:", OUTPUT_DIR)
    print("Total copied images:", total_images)
    print("Original raw data and processed data were not changed.")


if __name__ == "__main__":
    main()
