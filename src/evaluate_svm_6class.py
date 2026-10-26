
from pathlib import Path
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from extract_features import load_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load the newly trained six-class model
MODEL_PATH = PROJECT_DIR / "models" / "svm_model_6class.joblib"

CLASSES = [
    "hibiscus",
    "lotus",
    "marigold",
    "rose",
    "rose_periwinkle",
    "unknown",
]


def main():
    print("Loading six-class SVM model...")
    model = joblib.load(MODEL_PATH)

    print("\nLoading test dataset...")
    X_test, y_test = load_dataset("test")

    print("\nTest feature matrix shape:", X_test.shape)
    print("Number of test images:", len(y_test))

    print("\nEvaluating six-class SVM...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=CLASSES,
            zero_division=0,
        )
    )

    print("\nConfusion matrix")
    print("Rows = actual classes; columns = predicted classes")
    print("Class order:", CLASSES)
    print(confusion_matrix(y_test, y_pred, labels=CLASSES))


if __name__ == "__main__":
    main()
