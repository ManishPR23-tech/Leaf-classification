# Leaf Classification Using Machine Learning and Deep Learning

A leaf-image classification project combining classical machine learning and deep learning to classify images into six categories: **Hibiscus, Lotus, Marigold, Rose, Rose Periwinkle, and Unknown (Togor)**.

The project explores image preprocessing, feature extraction, model training, evaluation, error analysis, and image-based prediction through a Streamlit application.

## Project Objectives

- Build a reproducible image-classification pipeline.
- Organize and explore a labeled image dataset.
- Extract image features and train classical machine-learning models.
- Train a convolutional neural network and evaluate a MobileNetV2 transfer-learning model.
- Compare predictions using accuracy, precision, recall, F1-score, and confusion matrices.
- Explore unknown-category classification and inspect model errors.
- Provide an interface for making predictions on leaf images.

## Dataset

The dataset was assembled from the Bangladeshi Plant Leaves Dataset (BPLD) and additional flower-leaf images. The original image files are stored locally and are not included in this Git repository.

The six model categories are:

- Hibiscus
- Lotus
- Marigold
- Rose
- Rose Periwinkle
- Unknown (Togor)

The source folders contain 2,400 files. Exact duplicate files are excluded within each class during dataset splitting, leaving 2,394 unique source images.

| Split | Images |
|---|---:|
| Training | 1,675 |
| Validation | 359 |
| Testing | 360 |
| **Total** | **2,394** |

Each test class contains 60 images. The rose class has fewer training and validation images because duplicate source files were excluded.

The `unknown` category represents Togor images used as an unknown-category training class. It should not be interpreted as a guarantee that the model can recognize every plant species absent from its training data.

## Models

The repository contains implementations and evaluation scripts for several approaches:

1. **SVM:** A classical baseline using image pixel features and preprocessing.
2. **Random Forest:** A tree-based ensemble classifier.
3. **HOG-SVM:** An SVM trained using Histogram of Oriented Gradients features.
4. **Tuned SVM:** An SVM configuration selected using validation performance.
5. **CNN:** A convolutional neural network for image classification.
6. **MobileNetV2:** A deep-learning image classifier evaluated on the six-category dataset.

Saved models and supporting files are stored in `models/` locally. Availability depends on which model artifacts have been downloaded or generated.

## Results

### Six-class MobileNetV2 evaluation

The saved MobileNetV2 model was evaluated on the existing test dataset using `src/evaluate_mobilenet_6class.py`.

| Metric | Result |
|---|---:|
| Test images | 360 |
| Correct predictions | 357 |
| Incorrect predictions | 3 |
| **Test accuracy** | **99.17%** |

Per-class performance:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Hibiscus | 0.98 | 0.98 | 0.98 |
| Lotus | 1.00 | 0.98 | 0.99 |
| Marigold | 1.00 | 1.00 | 1.00 |
| Rose | 1.00 | 1.00 | 1.00 |
| Rose Periwinkle | 1.00 | 0.98 | 0.99 |
| Unknown | 0.97 | 1.00 | 0.98 |
| **Overall accuracy** | | | **0.99** |

The confusion matrix showed three incorrect predictions: one hibiscus image, one lotus image, and one rose-periwinkle image were misclassified.

These results describe the current test split and do not guarantee equivalent performance on images from different environments, cameras, lighting conditions, or unseen plant species. The unknown-category results specifically measure performance on the included Togor examples.

The repository also contains older three-class model-comparison results. Those results are not directly comparable with the six-class MobileNetV2 result because they use a different classification task.

## Project Structure

```text
Leaf-classification/
├── data/
│   ├── raw/
│   ├── unknown_raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── model_comparison.csv
├── src/
│   └── ... preprocessing, training,
│       evaluation, and visualization scripts
├── app.py
└── README.md
```

The exact files present locally may vary. Dataset images and some trained models are not tracked by Git.

## Requirements

- Python 3.11
- NumPy
- pandas
- Matplotlib
- scikit-learn
- TensorFlow / Keras
- Pillow
- Streamlit for the web application

Additional libraries may be required by individual scripts.

## Setup

Clone the repository:

```powershell
git clone https://github.com/ManishPR23-tech/Leaf-classification.git
cd Leaf-classification
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required libraries:

```powershell
python -m pip install numpy pandas matplotlib scikit-learn tensorflow pillow streamlit joblib scikit-image
```

Obtain the dataset separately and arrange it according to the directory structure expected by the scripts. The preprocessing and evaluation scripts contain the expected class names and paths.

## Running the Application

From the repository root, activate the virtual environment and run:

```powershell
streamlit run app.py
```

The application requires the relevant trained model files and any supporting files referenced by `app.py`.

## Running Evaluations

To evaluate the saved six-class MobileNetV2 model:

```powershell
python src/evaluate_mobilenet_6class.py
```

To evaluate the saved six-class SVM:

```powershell
python src/evaluate_svm_6class.py
```

These commands evaluate existing models; they do not retrain them.

Other scripts in `src/` support feature extraction, model evaluation, dataset inspection, and visualization. Check each script's dependencies before running it.

## Limitations

- The dataset covers a limited set of leaf categories.
- The `unknown` class represents the Togor examples used in this project, not every possible unseen species.
- Model performance may change with image quality, background, lighting, and acquisition conditions.
- High accuracy on one test split does not establish generalization to all real-world images.
- The older three-class and newer six-class results use different tasks and should not be compared directly.
- Dataset files and some model artifacts are excluded from the Git repository.

## Future Work

- Evaluate performance on independent images collected under different conditions.
- Improve rejection of genuinely unfamiliar species beyond the trained unknown category.
- Inspect failure cases and test robustness to backgrounds, lighting, and image quality.
- Compare models on the same six-class split using consistent metrics.
- Improve reproducibility through documented dependencies, dataset metadata, and evaluation records.

## Repository

GitHub: https://github.com/ManishPR23-tech/Leaf-classification
