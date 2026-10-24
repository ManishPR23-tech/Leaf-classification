
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

SEED = 42
tf.keras.utils.set_random_seed(SEED)


# --------------------------------------------------
# 3. Dataset settings
# --------------------------------------------------

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
CLASS_NAMES = ["hibiscus", "marigold", "rose"]

print("Loading training dataset...")

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

print("\nLoading validation dataset...")

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# --------------------------------------------------
# 4. Improve data-loading performance
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# --------------------------------------------------
# 5. Build the augmented CNN
# --------------------------------------------------

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Random transformations during training
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),

    # Normalize pixel values to the range 0–1
    layers.Rescaling(1.0 / 255),

    # Convolutional feature extraction
    layers.Conv2D(
        16, 3, activation="relu", padding="same"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        32, 3, activation="relu", padding="same"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64, 3, activation="relu", padding="same"
    ),
    layers.MaxPooling2D(),

    # Classification
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(CLASS_NAMES), activation="softmax"),
])


# --------------------------------------------------
# 6. Compile the model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# --------------------------------------------------
# 7. Save the augmented model separately
# --------------------------------------------------

MODEL_PATH = MODEL_DIR / "cnn_augmented_best.keras"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
]


# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

print("\nStarting augmented CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
)


# --------------------------------------------------
# 9. Save training history
# --------------------------------------------------

history_df = pd.DataFrame(history.history)

HISTORY_PATH = REPORT_DIR / "cnn_augmented_training_history.csv"

history_df.to_csv(HISTORY_PATH, index=False)

print("\nTraining finished.")
print("Best augmented model saved to:", MODEL_PATH)
print("Training history saved to:", HISTORY_PATH)
