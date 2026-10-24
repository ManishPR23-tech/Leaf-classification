
from pathlib import Path
from collections import defaultdict
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed_clean"

SPLITS = ["train", "val", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hash_to_files = defaultdict(list)

for split in SPLITS:
    split_dir = DATA_DIR / split

    for path in split_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_to_files[digest].append((split, path))

cross_split_groups = 0

for files in hash_to_files.values():
    splits_found = {split for split, _ in files}

    if len(splits_found) > 1:
        cross_split_groups += 1
        print("\nDuplicate across splits:")

        for split, path in files:
            print(f"  {split}: {path.relative_to(PROJECT_DIR)}")

print("\nTotal unique image contents:", len(hash_to_files))
print("Duplicate groups crossing splits:", cross_split_groups)
