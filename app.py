
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "mobilenet_6class_best.keras"
)

IMAGE_SIZE = (160, 160)

# This order MUST match the order used during training.
CLASS_NAMES = [
    "hibiscus",
    "lotus",
    "marigold",
    "rose",
    "rose_periwinkle",
    "unknown",
]


# ==================================================
# 2. WEBSITE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Leaf Classification",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Leaf Classification")

st.write(
    "Upload a flower photograph to predict its category "
    "using a MobileNetV2 image-classification model."
)

st.info(
    "The model recognizes six categories: Hibiscus, Lotus, "
    "Marigold, Rose, Rose Periwinkle, and Unknown."
)

st.caption(
    "This is an image-classification model. Its prediction "
    "is not a guaranteed botanical identification."
)


# ==================================================
# 3. LOAD THE TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    """Load the trained model once and reuse it."""
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.is_file():
    st.error(
        "The trained model was not found.\n\n"
        f"Expected location: {MODEL_PATH}\n\n"
        "Check that the model file exists inside your project's "
        "'models' folder."
    )
    st.stop()


# ==================================================
# 4. IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Choose a flower photograph",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear photograph of one flower or plant.",
)


# ==================================================
# 5. PROCESS IMAGE AND PREDICT
# ==================================================

if uploaded_file is not None:

    try:
        # Open the image and correct its orientation if needed.
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(
            image,
            caption="Uploaded photograph",
            width="stretch",
        )

        if st.button(
            "Predict Species",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner("Analyzing your image..."):

                # Load the saved six-class model.
                model = load_model()

                # Resize to the dimensions used during training.
                resized_image = image.resize(
                    IMAGE_SIZE,
                    Image.Resampling.BILINEAR,
                )

                # Convert the image to a float32 NumPy array.
                # Pixel values remain in the 0–255 range.
                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32,
                )

                # Add the batch dimension:
                # (160, 160, 3) -> (1, 160, 160, 3)
                image_array = np.expand_dims(
                    image_array,
                    axis=0,
                )

                # The saved model contains its own Rescaling layer.
                # Do NOT divide by 255 or scale to [-1, 1] here.
                probabilities = model.predict(
                    image_array,
                    verbose=0,
                )[0]

                # Check that the output matches the six classes.
                if len(probabilities) != len(CLASS_NAMES):
                    raise ValueError(
                        "The model output does not match the six "
                        "configured class names. Check that the "
                        "correct six-class model is being loaded."
                    )

                # Find the class with the highest predicted score.
                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = CLASS_NAMES[predicted_index]

                confidence = float(
                    probabilities[predicted_index]
                )


            # ==============================================
            # 6. DISPLAY THE MAIN PREDICTION
            # ==============================================

            st.subheader("Prediction")

            display_name = predicted_class.replace(
                "_", " "
            ).title()

            st.success(
                f"Predicted category: {display_name}"
            )

            st.metric(
                label="Model confidence",
                value=f"{confidence:.2%}",
            )

            # Explain the meaning of the unknown class.
            if predicted_class == "unknown":
                st.warning(
                    "This image most closely matches the "
                    "'Unknown' category in the training dataset, "
                    "which currently contains Togor images. "
                    "This does not confirm the plant's identity "
                    "or mean that every unfamiliar species will "
                    "be classified as unknown."
                )
            else:
                st.caption(
                    "The prediction is based on visual patterns "
                    "learned from the training images. Verify the "
                    "species independently if accurate identification "
                    "is important."
                )


            # ==============================================
            # 7. DISPLAY ALL SIX CLASS SCORES
            # ==============================================

            st.subheader("Scores for Each Category")

            for index, class_name in enumerate(CLASS_NAMES):

                score = float(probabilities[index])

                display_class_name = class_name.replace(
                    "_", " "
                ).title()

                st.write(
                    f"**{display_class_name}** — {score:.2%}"
                )

                st.progress(
                    max(0.0, min(score, 1.0))
                )


            # ==============================================
            # 8. TECHNICAL DETAILS
            # ==============================================

            with st.expander("Technical details"):

                st.write(
                    "**Model file:**",
                    str(MODEL_PATH),
                )

                st.write(
                    "**Input shape:**",
                    tuple(image_array.shape),
                )

                st.write(
                    "**Input pixel range:**",
                    f"{image_array.min():.1f} to "
                    f"{image_array.max():.1f}",
                )

                st.write(
                    "**Number of classes:**",
                    len(CLASS_NAMES),
                )

                st.write(
                    "**Class order:**",
                    CLASS_NAMES,
                )

                st.write(
                    "**Raw prediction scores:**",
                    probabilities.tolist(),
                )


    # ==================================================
    # 9. ERROR HANDLING
    # ==================================================

    except (UnidentifiedImageError, OSError):
        st.error(
            "The uploaded file could not be opened as an image. "
            "Please try a different JPG, JPEG, PNG, or WebP image."
        )

    except Exception as error:
        st.error(
            f"Something went wrong while making the prediction: "
            f"{error}"
        )
