
from pathlib import Path
import joblib

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from extract_features import load_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading training data...")
X_train, y_train = load_dataset("train")

print("Loading validation data...")
X_val, y_val = load_dataset("val")

# Test a small number of configurations.
configurations = [
    {"components": 50, "C": 1, "gamma": "scale"},
    {"components": 100, "C": 1, "gamma": "scale"},
    {"components": 150, "C": 1, "gamma": "scale"},
    {"components": 100, "C": 10, "gamma": "scale"},
    {"components": 100, "C": 10, "gamma": 0.001},
]

best_accuracy = -1
best_model = None
best_config = None

for config in configurations:
    print("\nTesting:", config)

    model = make_pipeline(
        StandardScaler(),
        PCA(
            n_components=config["components"],
            svd_solver="randomized",
            random_state=42,
        ),
        SVC(
            kernel="rbf",
            C=config["C"],
            gamma=config["gamma"],
        ),
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    print(f"Validation accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_config = config

print("\n========== BEST CONFIGURATION ==========")
print("Configuration:", best_config)
print(f"Best validation accuracy: {best_accuracy:.4f}")

print("\nValidation classification report:")
print(
    classification_report(
        y_val,
        best_model.predict(X_val),
        labels=["hibiscus", "marigold", "rose"],
        zero_division=0,
    )
)

output_path = MODEL_DIR / "svm_tuned_model.joblib"
joblib.dump(best_model, output_path)

print("\nBest model saved to:", output_path)
