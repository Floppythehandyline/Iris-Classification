# Iris Flower Classification Project

This project demonstrates the classification of Iris flowers using two popular Machine Learning algorithms: **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**.

## 📋 Project Overview

The project uses the classic Iris dataset to classify flowers into three species: Setosa, Versicolor, and Virginica. It includes data preprocessing, hyperparameter tuning, and detailed evaluation metrics.

### Key Features
- **Data Preprocessing**: Handling feature scaling (Standardization).
- **Hyperparameter Tuning**: Finding optimal `K` for KNN and `C`/`Kernel` for SVM using a validation set.
- **Evaluation**: Calculating Accuracy, Precision, and Recall on an unseen Test set.
- **Visualization**: SVM decision boundary visualization using PCA (Principal Component Analysis).

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed.

### Installation

1. Clone or download this repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🛠️ How to Run

### 1. KNN Classification
Run the K-Nearest Neighbors implementation:
```bash
python knn_main.py
```
This script will:
- Tune the `K` value from 1 to 15.
- Display validation accuracy for each `K`.
- Evaluate the best `K` on the test set.

### 2. SVM Classification
Run the Support Vector Machine implementation:
```bash
python svm_main.py
```
This script will:
- Tune `C` and `Kernel` parameters.
- Evaluate the final model on the test set.
- Generate and display a **Decision Boundary Plot** using PCA.

## 📁 File Structure

- `knn_main.py`: Main script for KNN classification.
- `svm_main.py`: Main script for SVM classification.
- `requirements.txt`: List of required Python libraries.
- `รายงาน_...pdf`: Technical reports (Thai language).

## 📊 Results Summary

Both algorithms perform exceptionally well on the Iris dataset, typically achieving:
- **Accuracy**: >95%
- **Precision (Macro)**: >95%
- **Recall (Macro)**: >95%

Visualization in `svm_main.py` provides clear insights into how the SVM model separates different species in a 2D projected space.
