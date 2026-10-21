
from pathlib import Path
from collections import Counter
from utils.preprocessing import preprocess_image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_images = 0
failed_images = []
class_counts = Counter()

for split in ["train", "val", "test"]:
    split_dir = DATA_DIR / split

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total_images += 1
            class_counts[(split, class_dir.name)] += 1

            try:
                result = preprocess_image(str(image_path))

                assert result.shape == (224, 224, 3)
                assert result.dtype.name == "float32"
                assert result.min() >= 0.0
                assert result.max() <= 1.0

            except Exception as error:
                failed_images.append((str(image_path), str(error)))

print("Total images checked:", total_images)
print("\nImages per split and class:")

for (split, class_name), count in sorted(class_counts.items()):
    print(f"{split:5} | {class_name:10} | {count}")

print("\nFailed images:", len(failed_images))

for image_path, error in failed_images[:10]:
    print(image_path)
    print("Error:", error)

if total_images > 0 and not failed_images:
    print("\nSUCCESS: All images passed preprocessing checks!")
else:
    print("\nCHECK REQUIRED: Some images failed or no images were found.")
