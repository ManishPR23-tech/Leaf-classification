
from pathlib import Path
import zipfile
import shutil

PROJECT_DIR = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_DIR / "flower.zip"
RAW_DIR = PROJECT_DIR / "data" / "raw"

NEW_CLASSES = ["lotus", "rose_periwinkle"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        for class_name in NEW_CLASSES:
            members = [
                name for name in archive.namelist()
                if name.startswith(f"flower/{class_name}/")
                and Path(name).suffix.lower() in IMAGE_EXTENSIONS
                and not name.endswith("/")
            ]

            if len(members) != 400:
                raise ValueError(
                    f"Expected 400 images for {class_name}, "
                    f"but found {len(members)}. Nothing extracted."
                )

        for class_name in NEW_CLASSES:
            destination = RAW_DIR / class_name
            destination.mkdir(parents=True, exist_ok=True)

            members = [
                name for name in archive.namelist()
                if name.startswith(f"flower/{class_name}/")
                and Path(name).suffix.lower() in IMAGE_EXTENSIONS
                and not name.endswith("/")
            ]

            copied = 0

            for member in members:
                filename = Path(member).name
                output_path = destination / filename

                if output_path.exists():
                    raise FileExistsError(
                        f"Destination already exists: {output_path}. "
                        "Stop and inspect before continuing."
                    )

                with archive.open(member) as source:
                    with output_path.open("wb") as target:
                        shutil.copyfileobj(source, target)

                copied += 1

            print(f"{class_name}: extracted {copied} images")

    print("\nNew classes added successfully.")


if __name__ == "__main__":
    main()
