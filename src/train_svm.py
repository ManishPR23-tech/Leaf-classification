
from pathlib import Path
import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from extract_features import load_dataset

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load the datasets
print("Loading training data...")
X_train, y_train = load_dataset("train")

print("Loading validation data...")
X_val, y_val = load_dataset("val")

print("\nOriginal training shape:", X_train.shape)
print("Original validation shape:", X_val.shape)

# Build a memory-conscious SVM pipeline
model = make_pipeline(
    StandardScaler(),
    PCA(n_components=100, svd_solver="randomized", random_state=42),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)

print("\nTraining SVM...")
model.fit(X_train, y_train)

# Evaluate on validation data
print("\nEvaluating on validation data...")
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_val, y_pred, zero_division=0))

# Save the trained model
model_path = MODEL_DIR / "svm_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
