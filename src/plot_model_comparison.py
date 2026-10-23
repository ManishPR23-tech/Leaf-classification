
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Find the project and report folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

CSV_PATH = REPORT_DIR / "model_comparison.csv"
OUTPUT_PATH = FIGURE_DIR / "model_accuracy_comparison.png"

# Create the figures folder if it doesn't exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Read the model results
df = pd.read_csv(CSV_PATH)

# Convert accuracy from proportions to percentages
df["test_accuracy_percent"] = df["test_accuracy"] * 100

# Plot the results
plt.figure(figsize=(10, 6))
bars = plt.bar(
    df["model"],
    df["test_accuracy_percent"],
)

plt.title("Test Accuracy Comparison of Leaf Classification Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20, ha="right")

# Display the accuracy above each bar
for bar, accuracy in zip(bars, df["test_accuracy_percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Chart saved to: {OUTPUT_PATH}")
