
from pathlib import Path

import numpy as np
import tensorflow as tf


# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "data" / "processed" / "train"
VALIDATION_DIR = PROJECT_DIR / "data" / "processed" / "val"

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "mobilenet_6class_best.keras"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

KNOWN_CLASSES = [
    "hibiscus",
    "lotus",
    "marigold",
    "rose",
    "rose_periwinkle",
]


# ==========================================
# 2. CHECK FILES AND FOLDERS
# ==========================================

print("Checking project files...")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

for folder in [TRAIN_DIR, VALIDATION_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Required folder not found: {folder}"
        )

for class_name in KNOWN_CLASSES:
    for folder in [TRAIN_DIR, VALIDATION_DIR]:
        class_dir = folder / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}"
            )

print("All required files and folders were found.")


# ==========================================
# 3. LOAD THE MODEL AND FEATURE EXTRACTOR
# ==========================================

print("\nLoading MobileNetV2 model...")

classifier = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

feature_model = tf.keras.Model(
    inputs=classifier.inputs,
    outputs=classifier.layers[-2].output,
)

print("Model loaded successfully.")
print("Feature output shape:", feature_model.output.shape)


# ==========================================
# 4. FUNCTION TO EXTRACT FEATURES
# ==========================================

def extract_features(dataset, dataset_name):
    features_list = []
    labels_list = []

    print(f"\nExtracting {dataset_name} features...")

    for batch_number, (images, labels) in enumerate(
        dataset,
        start=1,
    ):
        features = feature_model(
            images,
            training=False,
        ).numpy()

        features_list.append(features)
        labels_list.append(labels.numpy())

        print(
            f"{dataset_name}: processed batch "
            f"{batch_number}"
        )

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


# ==========================================
# 5. LOAD TRAINING AND VALIDATION DATA
# ==========================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=KNOWN_CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nLoading validation dataset...")

validation_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        labels="inferred",
        label_mode="int",
        class_names=KNOWN_CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
)


# ==========================================
# 6. EXTRACT FEATURES
# ==========================================

train_features, train_labels = extract_features(
    train_dataset,
    "Training",
)

val_features, val_labels = extract_features(
    validation_dataset,
    "Validation",
)

print("\nFeature extraction completed.")

print("Training feature shape:", train_features.shape)
print("Validation feature shape:", val_features.shape)


# ==========================================
# 7. CALCULATE CLASS CENTROIDS
# ==========================================

print("\nCalculating class centroids...")

centroids = {}

for class_index, class_name in enumerate(KNOWN_CLASSES):
    class_features = train_features[
        train_labels == class_index
    ]

    if len(class_features) == 0:
        raise ValueError(
            f"No training features for {class_name}"
        )

    centroids[class_index] = np.mean(
        class_features,
        axis=0,
    )

    print(
        f"{class_name}: centroid calculated from "
        f"{len(class_features)} images"
    )


# ==========================================
# 8. MEASURE DISTANCE TO THE CORRECT CLASS
# ==========================================

def euclidean_distance(features, centroid):
    return np.linalg.norm(
        features - centroid,
        axis=1,
    )


print("\nCalculating validation distances...")

val_distances = np.zeros(len(val_labels), dtype=float)

for class_index, class_name in enumerate(KNOWN_CLASSES):
    mask = val_labels == class_index

    val_distances[mask] = euclidean_distance(
        val_features[mask],
        centroids[class_index],
    )


# ==========================================
# 9. SUMMARIZE DISTANCES BY CLASS
# ==========================================

print("\n" + "=" * 55)
print("VALIDATION DISTANCE RESULTS")
print("=" * 55)

for class_index, class_name in enumerate(KNOWN_CLASSES):
    class_distances = val_distances[
        val_labels == class_index
    ]

    if len(class_distances) == 0:
        continue

    print(f"\nClass: {class_name}")
    print("Images:", len(class_distances))
    print("Mean distance:", round(float(np.mean(class_distances)), 4))
    print("Median distance:", round(float(np.median(class_distances)), 4))
    print("90th percentile:", round(float(np.percentile(class_distances, 90)), 4))
    print("95th percentile:", round(float(np.percentile(class_distances, 95)), 4))
    print("Maximum distance:", round(float(np.max(class_distances)), 4))


# ==========================================
# 10. OVERALL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("OVERALL VALIDATION SUMMARY")
print("=" * 55)

print("Training images:", len(train_labels))
print("Validation images:", len(val_labels))
print("Feature dimensions:", int(np.prod(train_features.shape[1:])))
print("All distances finite:", bool(np.all(np.isfinite(val_distances))))

print(
    "\nThis script measures distances for known validation "
    "images only. It does not yet classify unfamiliar leaves."
)
