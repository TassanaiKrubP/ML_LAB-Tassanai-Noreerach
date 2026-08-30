import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

# data_loader.py is one level up, shared with regression/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from data_loader import load_data, to_features, filter_roles, SKILL_COLUMNS
from model import (train_model, train_2d_model,
                   predict_model, predict_proba_model)
from evaluate import (evaluate_model, plot_roc_curve,
                      plot_decision_boundary, plot_samples)

CSV_PATH = os.path.join(ROOT_DIR, "fifa_players_clean.csv")
OTHERS_DIR = os.path.join(ROOT_DIR, "others_dir")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CLASSES = ["Midfielder", "Attacker"]   # 0 = Midfielder, 1 = Attacker
TEST_SIZE = 0.2
PCA_COMPONENTS = 10
C = 1.0

# Two skills that separate the classes best, for the boundary plot
BOUNDARY_FEATURES = ["finishing", "long_passing"]


def main():

    print("--" * 30)
    print("Classification: ATTACKER vs MIDFIELDER")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Read the spreadsheet
    print("\n[Step 1] Reading fifa_players_clean.csv...")
    skills, meta = load_data(CSV_PATH, OTHERS_DIR)
    print(f"Rows (all roles): {len(skills)}")

    # Step 2: Preparing classification data
    print("\n[Step 2] Preparing classification data...")
    skills, meta, _ = filter_roles(skills, meta, CLASSES)

    X = to_features(skills)
    y = (meta["role"] == "Attacker").to_numpy().astype(int)
    names = meta["name"].to_numpy()
    positions = meta["main_position"].to_numpy()

    print(f"Rows (2 classes): {len(X)}")
    for label, name in enumerate(CLASSES):
        count = (y == label).sum()
        print(f"  {name:<9}: {count}  ({count / len(y) * 100:.1f}%)")
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    index = np.arange(len(X))
    X_train, X_test, y_train, y_test, _, test_index = train_test_split(
        X, y, index, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Step 4: Train
    print(f"\n[Step 4] Training logistic regression "
          f"(PCA {PCA_COMPONENTS} components)...")
    model = train_model(X_train, y_train, PCA_COMPONENTS, C)
    print("Training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)
    probabilities = predict_proba_model(model, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, CLASSES,
                   os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    plot_roc_curve(y_test, probabilities, CLASSES[1],
                   os.path.join(OUTPUT_DIR, "roc_curve.png"))

    plot_samples(names[test_index], positions[test_index],
                 y_test, predictions, CLASSES,
                 os.path.join(OUTPUT_DIR, "role_samples.png"))

    # Step 7: Decision boundary, 2 features only
    print("\n[Step 7] Drawing decision boundary...")
    columns = [SKILL_COLUMNS.index(f) for f in BOUNDARY_FEATURES]

    # Raw 0-100 values here, so the axes read as real skill points
    raw = skills[:, columns].astype(float)
    raw_train, raw_test, y2_train, y2_test = train_test_split(
        raw, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    model_2d = train_2d_model(raw_train, y2_train, C)
    accuracy_2d = (model_2d.predict(raw_test) == y2_test).mean()
    print(f"Accuracy with only {BOUNDARY_FEATURES}: "
          f"{accuracy_2d * 100:.2f}%")

    plot_decision_boundary(model_2d, raw_test, y2_test,
                           BOUNDARY_FEATURES, CLASSES,
                           os.path.join(OUTPUT_DIR,
                                        "decision_boundary.png"))


if __name__ == "__main__":
    main()
