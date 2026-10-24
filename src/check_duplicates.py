
from pathlib import Path
from collections import defaultdict
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

SPLITS = ["train", "val", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Map each file's content hash to its locations.
hash_to_files = defaultdict(list)

for split in SPLITS:
    split_dir = DATA_DIR / split

    for path in split_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_to_files[digest].append(path)

# Report files with identical contents.
duplicates = {
    digest: paths
    for digest, paths in hash_to_files.items()
    if len(paths) > 1
}

print("Total unique file contents:", len(hash_to_files))
print("Groups of identical files:", len(duplicates))

cross_split_groups = 0

for paths in duplicates.values():
    splits_found = {path.parts[-3] for path in paths}

    if len(splits_found) > 1:
        cross_split_groups += 1
        print("\nIdentical files across splits:")
        for path in paths:
            print(" ", path.relative_to(PROJECT_DIR))

print("\nDuplicate groups crossing splits:", cross_split_groups)
