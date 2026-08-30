# ML-03-Regression & Classification

Explore and experiment with regression and classification using FIFA player
attribute data. This module uses the same dataset for two machine learning
tasks: overall rating prediction using regression and player role
classification.

The workflow starts by loading and preparing the dataset, then applies
StandardScaler and PCA for preprocessing and dimensionality reduction. The
processed features are used to train Ridge Regression for rating prediction
and Logistic Regression for role classification. Finally, each model is
evaluated using appropriate performance metrics and visualizations.

# Data

Dataset from Kaggle:

FIFA Players Dataset

The raw file was cleaned before use. Removed: `height_cm` (broken unit
conversion, only 20 distinct values), the four national team columns (95%
missing), `release_clause_euro` (correlates 0.994 with `value_euro`, a
leakage risk), 7 rows with invalid `body_type`, and 255 rows with missing
`value_euro` / `wage_euro`. Added: `main_position` and `role`.

Result: 17,692 rows, 47 columns, no missing values.

# Structure

```text

ML-03-Regression-Classification/
│
├── fifa_players_clean.csv      # dataset
├── data_loader.py              # read CSV: all
├── main.py                     # run all
│
├── others_dir/
│   ├── skills.npy
│   └── meta.csv
│
├── regression/
│   ├── main.py
│   ├── model.py                # StandardScaler → PCA → Ridge
│   ├── evaluate.py             # MAE, RMSE, R², graph
│   └── outputs/
│       ├── regression_results.png
│       ├── rating_samples.png
│       └── simple_vs_multiple.png
│
├── classification/
│   ├── main.py
│   ├── model.py                # StandardScaler → PCA → LogisticRegression
│   ├── evaluate.py             # accuracy, report, confusion matrix, ROC
│   └── outputs/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── role_samples.png
│       └── decision_boundary.png
└── requirements.txt
```

# Run

```bash
pip install -r requirements.txt
python main.py                    # both tasks
python regression/main.py         # regression only
python classification/main.py     # classification only
```

# Results

| Task | Model | Metric |
|---|---|---|
| Regression | Simple Linear (1 feature) | R² 0.246, MAE 4.69 |
| Regression | Ridge + PCA (28 features) | R² 0.622, MAE 3.33, RMSE 4.28 |
| Classification | Logistic + PCA (28 features) | Accuracy 88.25%, AUC 0.948 |
| Classification | Logistic (2 features) | Accuracy 87.84% |

# Summary

This repository demonstrates a complete ML workflow for regression and
classification using FIFA player data. The regression task predicts a
player's overall rating using Ridge Regression, while the classification
task separates attackers from midfielders using Logistic Regression. Both
tasks include data preprocessing with StandardScaler and PCA, model
training, evaluation, and result visualization.
