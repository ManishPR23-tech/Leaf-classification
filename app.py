
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

# Must exactly match the class order used during training.
CLASS_NAMES = [
    "hibiscus",
    "lotus",
    "marigold",
    "rose",
    "rose_periwinkle",
    "unknown",
]

# Experimental rejection settings.
# These values have not been scientifically validated.
CONFIDENCE_THRESHOLD = 0.85
MINIMUM_MARGIN = 0.15

# Additional caution message for accepted predictions.
CAUTION_THRESHOLD = 0.95


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
    "Upload a leaf or plant photograph to classify it "
    "using a six-class MobileNetV2 model."
)

st.info(
    "Recognized categories: Hibiscus, Lotus, Marigold, "
    "Rose, Rose Periwinkle, and Unknown."
)

st.caption(
    "The Unknown category currently represents Togor images "
    "from the training dataset. It does not represent every "
    "possible unfamiliar plant species."
)


# ==================================================
# 3. LOAD THE TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    """Load and cache the trained model."""
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.is_file():
    st.error(
        f"Model file not found:\n\n{MODEL_PATH}\n\n"
        "Check that the six-class model exists in the models folder."
    )
    st.stop()


# ==================================================
# 4. IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Choose a leaf photograph",
    type=["jpg", "jpeg", "png", "webp"],
    help="For best results, use a clear image with a visible leaf.",
)


# ==================================================
# 5. PROCESS IMAGE AND PREDICT
# ==================================================

if uploaded_file is not None:

    try:
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

                model = load_model()

                # Resize to match the training dimensions.
                resized_image = image.resize(
                    IMAGE_SIZE,
                    Image.Resampling.BILINEAR,
                )

                # Keep pixel values in the 0–255 range.
                # The saved model applies Rescaling internally.
                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32,
                )

                # Shape: (160, 160, 3) -> (1, 160, 160, 3)
                image_array = np.expand_dims(
                    image_array,
                    axis=0,
                )

                probabilities = model.predict(
                    image_array,
                    verbose=0,
                )[0]

                if len(probabilities) != len(CLASS_NAMES):
                    raise ValueError(
                        "Model output does not match the six "
                        "configured classes."
                    )

                if not np.all(np.isfinite(probabilities)):
                    raise ValueError(
                        "The model returned invalid prediction scores."
                    )

                # Highest-scoring and second-highest-scoring classes.
                sorted_indices = np.argsort(probabilities)[::-1]

                predicted_index = int(sorted_indices[0])
                second_index = int(sorted_indices[1])

                top_confidence = float(
                    probabilities[predicted_index]
                )

                second_confidence = float(
                    probabilities[second_index]
                )

                margin = top_confidence - second_confidence

                raw_class = CLASS_NAMES[predicted_index]

                # Reject predictions with low confidence or a
                # small difference between the top two scores.
                is_uncertain = (
                    top_confidence < CONFIDENCE_THRESHOLD
                    or margin < MINIMUM_MARGIN
                )

                if is_uncertain:
                    predicted_class = "unknown"
                else:
                    predicted_class = raw_class

            # ==============================================
            # 6. DISPLAY THE MAIN PREDICTION
            # ==============================================

            st.subheader("Prediction")

            display_name = predicted_class.replace(
                "_", " "
            ).title()

            if is_uncertain:
                st.warning(
                    f"Prediction: {display_name}\n\n"
                    "The model's result did not meet the current "
                    "confidence or score-separation requirements. "
                    "The image has been marked Unknown/uncertain."
                )

            elif predicted_class == "unknown":
                st.warning(
                    "Prediction: Unknown\n\n"
                    "The image most closely matches the Unknown "
                    "category in the training dataset, currently "
                    "represented by Togor."
                )

            else:
                st.success(
                    f"Predicted category: {display_name}"
                )

                # Extra caution for accepted predictions below 95%.
                if top_confidence < CAUTION_THRESHOLD:
                    st.info(
                        "Caution: This prediction passed the current "
                        "rejection rules, but its model score is below "
                        "95%. Verify the plant identity independently, "
                        "especially if it may be an unfamiliar species."
                    )

            st.metric(
                label="Highest model score",
                value=f"{top_confidence:.2%}",
            )

            st.metric(
                label="Difference between top two scores",
                value=f"{margin:.2%}",
            )

            if is_uncertain:
                st.caption(
                    "This is a rejection by the website's rule. "
                    "It does not prove that the plant is unfamiliar."
                )
            else:
                st.caption(
                    "Model scores are not guaranteed to represent "
                    "the true probability that a prediction is correct."
                )

            # ==============================================
            # 7. DISPLAY ALL SIX MODEL SCORES
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
            # 8. SHOW THE REJECTION RULES
            # ==============================================

            with st.expander("Current rejection settings"):

                st.write(
                    f"Confidence threshold: "
                    f"{CONFIDENCE_THRESHOLD:.0%}"
                )

                st.write(
                    f"Minimum top-two score difference: "
                    f"{MINIMUM_MARGIN:.0%}"
                )

                st.write(
                    f"Extra caution threshold: "
                    f"{CAUTION_THRESHOLD:.0%}"
                )

                st.write(
                    "A prediction is rejected when its highest "
                    "score is below the confidence threshold OR "
                    "the difference between its top two scores "
                    "is below the minimum margin."
                )

                st.warning(
                    "These settings have not been validated on a "
                    "representative set of unfamiliar species. "
                    "An unfamiliar leaf may still receive a high score "
                    "and pass both rejection checks."
                )

            # ==============================================
            # 9. TECHNICAL DETAILS
            # ==============================================

            with st.expander("Technical details"):

                st.write("Model file:", str(MODEL_PATH))
                st.write("Input shape:", tuple(image_array.shape))

                st.write(
                    "Input pixel range:",
                    f"{image_array.min():.1f} to "
                    f"{image_array.max():.1f}",
                )

                st.write("Class order:", CLASS_NAMES)

                st.write(
                    "Raw prediction scores:",
                    probabilities.tolist(),
                )

                st.write("Top predicted class:", raw_class)
                st.write(
                    "Second predicted class:",
                    CLASS_NAMES[second_index],
                )
                st.write("Score margin:", f"{margin:.4f}")
                st.write("Rejected as uncertain:", is_uncertain)

    # ==================================================
    # 10. ERROR HANDLING
    # ==================================================

    except (UnidentifiedImageError, OSError):
        st.error(
            "The uploaded file could not be opened as an image. "
            "Try another JPG, JPEG, PNG, or WebP file."
        )

    except Exception as error:
        st.error(
            f"Something went wrong while making the prediction: "
            f"{error}"
        )
