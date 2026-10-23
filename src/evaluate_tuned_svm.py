
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from extract_features import load_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "svm_tuned_model.joblib"

CLASSES = ["hibiscus", "marigold", "rose"]

print("Loading tuned SVM model...")
model = joblib.load(MODEL_PATH)

print("\nLoading test dataset...")
X_test, y_test = load_dataset("test")

print("\nEvaluating tuned SVM on test data...")
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

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred, labels=CLASSES))
