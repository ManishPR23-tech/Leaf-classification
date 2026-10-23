# Leaf Classification Using Machine Learning and Deep Learning

A machine-learning project to classify leaf images into three flower species: **Hibiscus, Marigold, and Rose**.

The project compares classical machine-learning algorithms with a Convolutional Neural Network (CNN) to investigate how different approaches perform on leaf-image classification.

## Project Objectives

- Build a reproducible image-classification pipeline.
- Explore and prepare a labeled leaf-image dataset.
- Train and evaluate multiple classification models.
- Compare model performance using accuracy and confusion matrices.
- Identify classification errors and areas for improvement.

## Dataset

The project uses images from the BPLD (Bangladeshi Plant Leaves Dataset), selecting three flower classes:

- Hibiscus
- Marigold
- Rose

| Dataset split | Images per class | Total images |
|---|---:|---:|
| Training | 280 | 840 |
| Validation | 60 | 180 |
| Testing | 60 | 180 |
| **Total** | **400** | **1,200** |

The dataset is divided into training, validation, and testing subsets. The same split is used for the model comparisons.

**Dataset note:** The image files are stored locally and are not included in this Git repository. See the project setup instructions before running the scripts.

## Models

The project currently includes five approaches:

1. **Original SVM** — a baseline using PCA and raw pixel features.
2. **Random Forest** — a tree-based ensemble classifier.
3. **HOG-SVM** — uses Histogram of Oriented Gradients (HOG) image features with an SVM classifier.
4. **Tuned SVM** — evaluates selected hyperparameter configurations using validation accuracy.
5. **CNN** — a convolutional neural network trained on images resized to 128 × 128 pixels.

## Results

The following are the currently recorded test-set accuracies:

| Model | Test accuracy |
|---|---:|
| Original SVM | 85.00% |
| Random Forest | 85.00% |
| HOG-SVM | 80.56% |
| Tuned SVM | 85.00% |
| CNN | **85.56%** |

The CNN achieved the highest recorded test accuracy, correctly classifying 154 out of 180 test images.

These results are exploratory: the test set has been used to compare multiple models. Future model selection and tuning should rely on validation data, with a final untouched test set reserved for unbiased evaluation.

## Visualizations

The repository includes the following generated results:

- `reports/figures/model_accuracy_comparison.png` — comparison of test accuracy across the five models.
- `reports/figures/cnn_confusion_matrix.png` — CNN classification results by actual and predicted class.
- `reports/figures/tuned_svm_validation_confusion_matrix.png` — validation confusion matrix for the tuned SVM.

The CNN confusion matrix indicates that marigold is currently the most challenging class, particularly because some marigold images are predicted as rose.

## Project Structure

```text
Leaf-classification/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── model_comparison.csv
├── src/
│   ├── utils/
│   ├── check_image_integrity.py
│   ├── evaluate_cnn.py
│   ├── evaluate_model.py
│   ├── evaluate_random_forest.py
│   ├── evaluate_hog_svm.py
│   ├── evaluate_tuned_svm.py
│   ├── extract_features.py
│   ├── extract_hog_features.py
│   ├── plot_cnn_confusion_matrix.py
│   ├── plot_model_comparison.py
│   ├── plot_tuned_confusion_matrix.py
│   ├── split_dataset.py
│   ├── train_cnn.py
│   ├── train_random_forest.py
│   ├── train_svm.py
│   ├── train_hog_svm.py
│   └── tune_svm.py
└── README.md
```

Some generated files and folders are stored locally and may not be tracked by Git. The structure above describes the main project components.

## Requirements

- Python 3.11
- NumPy
- pandas
- Matplotlib
- scikit-learn
- TensorFlow / Keras
- Seaborn
- Pillow

Install the required libraries in your project's virtual environment. For example, in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scikit-learn tensorflow seaborn pillow
```

## Setup

1. Clone the repository:

   ```powershell
   git clone https://github.com/ManishPR23-tech/Leaf-classification.git
   cd Leaf-classification
   ```

2. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the required libraries using the command above.

4. Obtain the dataset separately and arrange the images according to the directory structure expected by the scripts:

   ```text
   data/processed/
   ├── train/
   │   ├── hibiscus/
   │   ├── marigold/
   │   └── rose/
   ├── val/
   │   ├── hibiscus/
   │   ├── marigold/
   │   └── rose/
   └── test/
       ├── hibiscus/
       ├── marigold/
       └── rose/
   ```

5. Run the scripts from the repository root using the virtual environment's Python interpreter. For example:

   ```powershell
   .\.venv\Scripts\python.exe src/train_cnn.py
   .\.venv\Scripts\python.exe src/evaluate_cnn.py
   .\.venv\Scripts\python.exe src/plot_model_comparison.py
   ```

**Note:** The training scripts may depend on generated features or trained models. Follow the appropriate preprocessing and training steps before running dependent evaluation scripts.

## Limitations

- The dataset contains only three flower classes.
- Results may depend on lighting, background, image quality, and variation within each class.
- Test accuracy alone does not establish that one model is significantly better than another.
- The dataset and trained model files are not included in the Git repository.

## Future Work

- Improve marigold-versus-rose discrimination.
- Investigate data augmentation and class weighting using validation data.
- Inspect misclassified images to understand common error patterns.
- Evaluate the final selected model on a fresh, untouched test set.
- Explore transfer learning with a pretrained image-classification model.
- Develop a simple interface for predicting the class of a new leaf image.

## Repository

GitHub: https://github.com/ManishPR23-tech/Leaf-classification
