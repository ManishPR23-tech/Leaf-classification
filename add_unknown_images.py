
from pathlib import Path
import zipfile
import shutil

PROJECT_DIR = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_DIR / "flower.zip"
OUTPUT_DIR = PROJECT_DIR / "data" / "unknown_raw" / "togor"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} is not empty. Nothing was extracted."
        )

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = [
            name for name in archive.namelist()
            if name.startswith("flower/togor/")
            and Path(name).suffix.lower() in IMAGE_EXTENSIONS
            and not name.endswith("/")
        ]

        if len(members) != 400:
            raise ValueError(
                f"Expected 400 Togor images, found {len(members)}."
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for member in members:
            destination = OUTPUT_DIR / Path(member).name

            if destination.exists():
                raise FileExistsError(f"File already exists: {destination}")

            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    print(f"Extracted {len(members)} Togor images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
