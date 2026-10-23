
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from extract_features import load_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "svm_tuned_model.joblib"
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

classes = ["hibiscus", "marigold", "rose"]

print("Loading tuned SVM...")
model = joblib.load(MODEL_PATH)

print("Loading validation data...")
X_val, y_val = load_dataset("val")
y_pred = model.predict(X_val)

cm = confusion_matrix(y_val, y_pred, labels=classes)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes,
)

display.plot(values_format="d")
plt.title("Tuned SVM — Validation Confusion Matrix")
plt.tight_layout()

output_path = REPORT_DIR / "tuned_svm_validation_confusion_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Confusion matrix saved to:", output_path)
