
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
CLASS_NAMES = ["hibiscus", "marigold", "rose"]

# Load datasets
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

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Load MobileNetV2 pretrained on ImageNet
base_model = keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze pretrained layers
base_model.trainable = False

# Build the classifier
inputs = keras.Input(shape=(160, 160, 3))

# MobileNetV2 expects input pixels scaled to [-1, 1]
x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Save the best model using validation accuracy
MODEL_PATH = MODEL_DIR / "mobilenet_best.keras"

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

# Train only the new classification head
print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
)

# Save training history
history_df = pd.DataFrame(history.history)

HISTORY_PATH = REPORT_DIR / "mobilenet_training_history.csv"
history_df.to_csv(HISTORY_PATH, index=False)

print("\nTraining finished.")
print("Best model saved to:", MODEL_PATH)
print("Training history saved to:", HISTORY_PATH)
