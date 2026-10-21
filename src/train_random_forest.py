
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from extract_features import load_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading training data...")
X_train, y_train = load_dataset("train")

print("Loading validation data...")
X_val, y_val = load_dataset("val")

print("\nTraining data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

print("\nEvaluating on validation data...")
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification report:")
print(classification_report(y_val, y_pred, zero_division=0))

model_path = MODEL_DIR / "random_forest_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
