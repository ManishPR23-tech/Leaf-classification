
from pathlib import Path
from PIL import Image
import warnings

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

checked = 0
unreadable = []
warning_files = []

print("Checking image integrity...")
print("Dataset:", DATA_DIR)

for image_path in sorted(DATA_DIR.rglob("*")):
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in VALID_EXTENSIONS:
        continue

    checked += 1

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            # Verify that the image can be decoded.
            with Image.open(image_path) as image:
                image.verify()

            # Reopen and fully load the image to detect truncated data.
            with Image.open(image_path) as image:
                image.load()

        if caught:
            messages = [str(item.message) for item in caught]
            warning_files.append((image_path, messages))

    except Exception as error:
        unreadable.append((image_path, str(error)))

print("\n========== RESULTS ==========")
print("Images checked:", checked)
print("Files with warnings:", len(warning_files))
print("Unreadable images:", len(unreadable))

if warning_files:
    print("\n--- Files that produced warnings ---")
    for path, messages in warning_files:
        print(f"\n{path}")
        for message in messages:
            print("  Warning:", message)

if unreadable:
    print("\n--- Unreadable images ---")
    for path, error in unreadable:
        print(f"\n{path}")
        print("  Error:", error)

if not warning_files and not unreadable:
    print("\nAll images passed without warnings.")
