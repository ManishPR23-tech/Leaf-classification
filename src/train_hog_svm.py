
from pathlib import Path

import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from extract_hog_features import load_hog_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading HOG training data...")
X_train, y_train = load_hog_dataset("train")

print("Loading HOG validation data...")
X_val, y_val = load_hog_dataset("val")

print("\nTraining feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale"),
)

print("\nTraining HOG-based SVM...")
model.fit(X_train, y_train)

print("\nEvaluating on validation data...")
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"\nHOG-SVM validation accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification report:")
print(classification_report(y_val, y_pred, zero_division=0))

model_path = MODEL_DIR / "hog_svm_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
