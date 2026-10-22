
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from extract_hog_features import load_hog_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "hog_svm_model.joblib"

print("Loading saved HOG-SVM model...")
model = joblib.load(MODEL_PATH)

print("Loading HOG test data...")
X_test, y_test = load_hog_dataset("test")

print("\nTest feature shape:", X_test.shape)
print("\nEvaluating HOG-SVM on test data...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nHOG-SVM test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

classes = ["hibiscus", "marigold", "rose"]

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=classes,
        zero_division=0,
    )
)

print("\nConfusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, y_pred, labels=classes))
