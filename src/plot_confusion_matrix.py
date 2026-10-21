
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from extract_features import load_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "svm_model.joblib"
OUTPUT_DIR = PROJECT_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

classes = ["hibiscus", "marigold", "rose"]

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading test dataset...")
X_test, y_test = load_dataset("test")
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=classes)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes,
)

display.plot(cmap="Blues", values_format="d")
plt.title("SVM Confusion Matrix - Test Dataset")
plt.tight_layout()

output_path = OUTPUT_DIR / "svm_confusion_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Confusion matrix saved to: {output_path}")

plt.show()
