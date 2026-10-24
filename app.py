
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "mobilenet_clean_best.keras"
)

IMAGE_SIZE = (160, 160)

# Must match the class order used during training
CLASS_NAMES = ["hibiscus", "marigold", "rose"]


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
    "Upload a photograph of a leaf to predict whether "
    "it belongs to hibiscus, marigold, or rose."
)

st.info(
    "This model recognizes three species: "
    "Hibiscus, Marigold, and Rose."
)


# ==================================================
# 3. LOAD THE TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    """Load the trained model and cache it."""
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


# ==================================================
# 4. UPLOAD AN IMAGE
# ==================================================

uploaded_file = st.file_uploader(
    "Choose a leaf photograph",
    type=["jpg", "jpeg", "png", "webp"],
)


# ==================================================
# 5. PROCESS THE IMAGE AND MAKE A PREDICTION
# ==================================================

if uploaded_file is not None:

    try:
        # Open the uploaded image and convert it to RGB
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded leaf image",
            width="stretch",
        )

        if st.button("Predict Species", type="primary"):

            with st.spinner("Analyzing your image..."):

                # Load the saved model
                model = load_model()

                # Resize the image to match training dimensions
                resized_image = image.resize(IMAGE_SIZE)

                # Convert the image to a NumPy array.
                # Pixel values remain in the 0–255 range.
                image_array = np.array(
                    resized_image,
                    dtype=np.float32,
                )

                # Add a batch dimension:
                # (160, 160, 3) -> (1, 160, 160, 3)
                #
                # IMPORTANT:
                # Do NOT divide by 127.5 or subtract 1 here.
                # The saved model already normalizes internally.
                image_array = np.expand_dims(
                    image_array,
                    axis=0,
                )

                # Generate prediction scores
                probabilities = model.predict(
                    image_array,
                    verbose=0,
                )[0]

                # Identify the species with the highest score
                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = CLASS_NAMES[predicted_index]

                confidence = float(
                    probabilities[predicted_index]
                )

            # ==========================================
            # 6. DISPLAY THE PREDICTION
            # ==========================================

            st.subheader("Prediction")

            st.success(
                f"Predicted species: {predicted_class.title()}"
            )

            st.metric(
                label="Model confidence",
                value=f"{confidence:.2%}",
            )

            # ==========================================
            # 7. DISPLAY SCORES FOR EACH SPECIES
            # ==========================================

            st.subheader("Scores for Each Species")

            for index, class_name in enumerate(CLASS_NAMES):

                score = float(probabilities[index])

                st.write(
                    f"**{class_name.title()}** — {score:.2%}"
                )

                st.progress(score)

            # ==========================================
            # 8. DIAGNOSTIC INFORMATION
            # ==========================================

            with st.expander("Technical details"):

                st.write("Model file:", str(MODEL_PATH))

                st.write(
                    "Input image shape:",
                    image_array.shape,
                )

                st.write(
                    "Input pixel range:",
                    f"{image_array.min():.1f} to "
                    f"{image_array.max():.1f}",
                )

                st.write(
                    "Raw prediction scores:",
                    probabilities.tolist(),
                )

            # ==========================================
            # 9. IMPORTANT LIMITATIONS
            # ==========================================

            st.caption(
                "Confidence scores are model outputs, not "
                "guarantees of correct identification."
            )

            st.warning(
                "This model only classifies hibiscus, "
                "marigold, and rose. It may give an incorrect "
                "prediction for unfamiliar species, unusual "
                "images, or photographs with difficult lighting."
            )

    except Exception as error:
        st.error(
            f"Could not process this image: {error}"
        )
