
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

# Locate the project and training images
PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "processed" / "train"

CLASSES = ["hibiscus", "rose", "marigold"]
random.seed(42)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for row, class_name in enumerate(CLASSES):
    class_dir = TRAIN_DIR / class_name

    images = [
        path for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    if not images:
        raise ValueError(f"No images found in {class_dir}")

    selected_images = random.sample(images, min(4, len(images)))

    for col, ax in enumerate(axes[row]):
        if col < len(selected_images):
            image_path = selected_images[col]

            with Image.open(image_path) as image:
                ax.imshow(image.convert("RGB"))

            ax.set_title(class_name.title())
        ax.axis("off")

plt.suptitle("Sample Images from the Leaf Classification Dataset")
plt.tight_layout()
plt.show()
