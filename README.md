# CT Slice Localization — ML Regression

Predicting the relative anatomical position of CT scan slices (0–100 scale) from 384 HOG image features, comparing XGBoost and deep learning approaches.

## Results

--- Comparison Table ---
| Model                    |     MAE |    RMSE |       R² | MAE Improvement vs Baseline   |
|:-------------------------|--------:|--------:|---------:|:------------------------------|
| XGBoost Baseline         | 2.8882  | 5.3562  | 0.9379   | 0.0%                          |
| XGBoost + Optuna         | 2.5808  | 5.1946  | 0.9416   | 10.6%                         |
| Best NN (CNN1D) + Optuna | 1.75505 | 3.19588 | 0.977903 | 39.2%                         |

> **10.6% MAE improvement** over baseline — ~4.6 cm average error on a full-body scan.

## Approaches

**Notebook 1 — Deep Learning** (`DNN_CNN_Regression.ipynb`)
- DNN, 1D CNN, and Hybrid (parallel DNN+CNN fusion) in PyTorch
- Huber Loss, AdamW, OneCycleLR, BatchNorm + Dropout
- Bayesian hyperparameter search via Optuna TPE (15 trials)

**Notebook 2 — XGBoost** (`Slice_Project.ipynb`)
- Gradient boosting with GPU acceleration (`tree_method="hist"`)
- Bayesian optimization via Optuna TPE (Up to 50 trials)
- Full EDA: patient coverage analysis, feature filtering, residuals

## Key Design Decisions

- **GroupShuffleSplit** by `patientId` — prevents data leakage between patients
- **3-stage feature reduction**: zero-variance → high correlation (>0.95) → low target relevance (<0.05) — reduces 384 → 289 features
- **1D CNN** exploits the spatial ordering of HOG features (neighboring features describe adjacent image regions)
- **Huber Loss (δ=5.0)** — robust to outlier errors at body extremities

## Stack

`Python` `PyTorch` `XGBoost` `Optuna` `scikit-learn` `pandas` `matplotlib`
