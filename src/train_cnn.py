
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Dataset settings
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

# Improve data-loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Build a small CNN
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(16, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(CLASS_NAMES), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Save the best model according to validation accuracy
MODEL_PATH = MODEL_DIR / "cnn_best.keras"

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

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
)
import pandas as pd

history_df = pd.DataFrame(history.history)

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

history_df.to_csv(REPORT_DIR / "cnn_training_history.csv", index=False)

print("\nTraining history saved to reports/cnn_training_history.csv")

print("\nTraining finished.")
print("Best model saved to:", MODEL_PATH)
