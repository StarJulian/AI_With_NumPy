# ML_From_Scratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.22+-orange.svg)

**Machine Learning Algorithms from Scratch using NumPy**

[English](./README.md) | [中文](./README_zh.md)

</div>

---

## Overview

A comprehensive machine learning library implemented from scratch using only NumPy. Each algorithm includes:
- **Training code**: Full implementation with gradient descent, coordinate descent, or EM algorithm
- **Inference code**: Prediction methods for both regression and classification
- **Demo examples**: Runnable examples with synthetic or real datasets

> ⚠️ **Disclaimer**: For educational purposes only. Not for commercial use.

---

## Structure

```
ML_From_Scratch/
├── __init__.py                # Package entry
├── training_framework.py      # Base classes for training
├── losses.py                  # Loss functions
│
├── linear_regression.py       # Linear Regression
├── logistic_regression.py     # Logistic Regression
├── decision_tree.py           # Decision Tree (ID3/CART)
├── random_forest.py           # Random Forest
├── svm.py                     # Support Vector Machine
├── naive_bayes.py             # Naive Bayes
├── kmeans.py                  # K-Means Clustering
├── knn.py                     # K-Nearest Neighbors
├── markov.py                  # Markov Chain
├── hidden_markov.py           # Hidden Markov Model
├── pca.py                     # Principal Component Analysis
├── adaboost.py                # AdaBoost
│
├── LICENSE                    # MIT License
└── README.md                  # English documentation
```

---

## Quick Start

### Requirements
- Python 3.9+
- NumPy 1.22+

```bash
pip install numpy matplotlib
```

### Run Demos

```bash
# Linear Regression
python ML/linear_regression.py

# Logistic Regression
python ML/logistic_regression.py

# Decision Tree
python ML/decision_tree.py

# K-Means
python ML/kmeans.py
```

---

## Algorithms

| Algorithm | File | Type | Method |
|-----------|------|------|--------|
| Linear Regression | `linear_regression.py` | Regression | Gradient Descent / Normal Equation |
| Logistic Regression | `logistic_regression.py` | Classification | Gradient Descent |
| Decision Tree | `decision_tree.py` | Classification | ID3 / CART |
| Random Forest | `random_forest.py` | Classification | Ensemble |
| SVM | `svm.py` | Classification | SMO |
| Naive Bayes | `naive_bayes.py` | Classification | Bayesian |
| K-Means | `kmeans.py` | Clustering | Lloyd's Algorithm |
| KNN | `knn.py` | Classification | Distance Weighted |
| Markov Chain | `markov.py` | Sequence | Transition Matrix |
| Hidden Markov | `hidden_markov.py` | Sequence | Baum-Welch / Viterbi |
| PCA | `pca.py` | Dimensionality Reduction | SVD |
| AdaBoost | `adaboost.py` | Ensemble | Boosting |

---

## Usage Example

```python
from ML import LinearRegression
import numpy as np

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression(input_dim=1, method='gd')
model.fit(X, y, learning_rate=0.01, max_iter=1000)

# Predict
X_test = np.array([[6], [7]])
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred}")  # [12, 14]
```

---

## License

MIT License - Educational use only
