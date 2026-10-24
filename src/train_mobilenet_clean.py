
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed_clean"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "mobilenet_clean_best.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 20

CLASS_NAMES = ["hibiscus", "marigold", "rose"]

SEED = 42


# --------------------------------------------------
# 2. Check the dataset folders
# --------------------------------------------------

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found: {VAL_DIR}"
    )

for class_name in CLASS_NAMES:
    if not (TRAIN_DIR / class_name).exists():
        raise FileNotFoundError(
            f"Missing training class folder: {class_name}"
        )

    if not (VAL_DIR / class_name).exists():
        raise FileNotFoundError(
            f"Missing validation class folder: {class_name}"
        )


# --------------------------------------------------
# 3. Load training and validation images
# --------------------------------------------------

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nClass names:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# --------------------------------------------------
# 4. Load the pretrained MobileNetV2 base
# --------------------------------------------------

base_model = keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet",
)

# Keep pretrained feature-extraction layers frozen
base_model.trainable = False


# --------------------------------------------------
# 5. Build the classification model
# --------------------------------------------------

inputs = keras.Input(shape=(160, 160, 3))

# The model receives pixel values in the [0, 255] range.
# Scale them to [-1, 1] inside the model.
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1,
)(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
)


# --------------------------------------------------
# 6. Compile the model
# --------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# --------------------------------------------------
# 7. Save the best model during training
# --------------------------------------------------

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
    verbose=1,
)


# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
    ],
)


# --------------------------------------------------
# 9. Display the best validation accuracy
# --------------------------------------------------

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print(
    "\nBest validation accuracy:",
    f"{best_val_accuracy:.2%}",
)

print("\nTraining finished.")
print("Best model location:", MODEL_PATH)
