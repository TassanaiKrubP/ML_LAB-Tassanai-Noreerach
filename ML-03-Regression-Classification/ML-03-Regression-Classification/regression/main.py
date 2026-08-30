import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

# data_loader.py is one level up, shared with classification/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from data_loader import load_data, to_features, SKILL_COLUMNS
from model import train_model, train_simple_model, predict_model
from evaluate import evaluate_model, plot_samples, compare_simple_multiple

CSV_PATH = os.path.join(ROOT_DIR, "fifa_players_clean.csv")
OTHERS_DIR = os.path.join(ROOT_DIR, "others_dir")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TEST_SIZE = 0.2
PCA_COMPONENTS = 15
ALPHA = 1.0

# Highest correlation with overall_rating among the real skills (0.51)
SIMPLE_FEATURE = "short_passing"


def main():

    print("--" * 30)
    print("Regression: predict OVERALL RATING from 28 skills")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Read the spreadsheet
    print("\n[Step 1] Reading fifa_players_clean.csv...")
    skills, meta = load_data(CSV_PATH, OTHERS_DIR)

    print(f"Rows         : {len(skills)}")
    print(f"Rating range : {meta['overall_rating'].min()} - "
          f"{meta['overall_rating'].max()}")
    print(f"Rating mean  : {meta['overall_rating'].mean():.1f}")

    # Step 2: Preprocessing
    print("\n[Step 2] Building feature matrix...")
    X = to_features(skills)
    y = meta["overall_rating"].to_numpy()
    names = meta["name"].to_numpy()
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    index = np.arange(len(X))
    X_train, X_test, y_train, y_test, _, test_index = train_test_split(
        X, y, index, test_size=TEST_SIZE, random_state=42
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Step 4: Train
    print(f"\n[Step 4] Training Ridge regression "
          f"(PCA {PCA_COMPONENTS} components)...")
    model = train_model(X_train, y_train, PCA_COMPONENTS, ALPHA)
    print("Training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    mae, rmse, r2 = evaluate_model(
        y_test, predictions,
        os.path.join(OUTPUT_DIR, "regression_results.png"))

    plot_samples(names[test_index], y_test, predictions,
                 os.path.join(OUTPUT_DIR, "rating_samples.png"))

    # Step 7: Simple vs Multiple, on the same split
    print("\n[Step 7] Comparing Simple vs Multiple regression...")
    column = SKILL_COLUMNS.index(SIMPLE_FEATURE)

    simple = train_simple_model(X_train[:, [column]], y_train)
    simple_predictions = predict_model(simple, X_test[:, [column]])

    from sklearn.metrics import r2_score, mean_absolute_error
    simple_r2 = r2_score(y_test, simple_predictions)
    simple_mae = mean_absolute_error(y_test, simple_predictions)

    compare_simple_multiple(
        [(f"Simple ({SIMPLE_FEATURE})", simple_r2, simple_mae),
         ("Multiple (28 skills)", r2, mae)],
        os.path.join(OUTPUT_DIR, "simple_vs_multiple.png"))


if __name__ == "__main__":
    main()
