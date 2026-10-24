
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# Find the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "train"

CLASS_NAMES = ["hibiscus", "marigold", "rose"]
IMAGES_PER_CLASS = 8
SEED = 42

random.seed(SEED)

fig, axes = plt.subplots(
    len(CLASS_NAMES),
    IMAGES_PER_CLASS,
    figsize=(16, 8)
)

for row, class_name in enumerate(CLASS_NAMES):
    class_dir = DATA_DIR / class_name

    image_paths = [
        path for path in class_dir.iterdir()
        if path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    ]

    selected_paths = random.sample(
        image_paths,
        min(IMAGES_PER_CLASS, len(image_paths))
    )

    for col in range(IMAGES_PER_CLASS):
        ax = axes[row, col]
        ax.axis("off")

        if col < len(selected_paths):
            path = selected_paths[col]

            try:
                with Image.open(path) as image:
                    ax.imshow(image.convert("RGB"))

                ax.set_title(
                    f"{class_name}\n{path.name[:16]}",
                    fontsize=7
                )

            except (OSError, UnidentifiedImageError):
                ax.set_title("Unreadable image", fontsize=8)

plt.suptitle(
    "Random Training Images: Hibiscus, Marigold and Rose",
    fontsize=14
)
plt.tight_layout()

OUTPUT_PATH = (
    PROJECT_DIR / "reports" / "figures"
    / "dataset_inspection.png"
)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
plt.show()

print(f"\nInspection image saved to:\n{OUTPUT_PATH}")
