import csv
import json
import os

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from knn_model import build_scaler, train_knn, predict_knn, sweep_k
from evaluate import (evaluate_model, plot_k_curve, report_confusions)

# Paths are resolved from this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = "outputs"

# 32 x 32 grayscale gives 1024 features per image. A larger size would give
# KNN more raw data but hurt it: as the number of dimensions grows, the
# distances between every pair of points become almost equal and "nearest
# neighbour" stops meaning anything. That is the curse of dimensionality.
IMG_SIZE = 32
TEST_SIZE = 0.2
MAX_PER_CLASS = 500      # keeps the classes balanced; None = use all images
K_VALUES = [3, 5, 7]     # required by the lab sheet
K_SWEEP = list(range(1, 32, 2))   # odd values only, to avoid tied votes


def main():

    print("--" * 30)
    print("KNN Image Classification: Aeroplanes vs Birds vs Drones")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes, paths = load_data(
        DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}  (samples, features per image)")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Step 4: Feature Scaling
    print("\n[Step 4] Standardizing features...")

    scaler = build_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    print(f"Before scaling: mean = {X_train.mean():.4f}, "
          f"std = {X_train.std():.4f}")
    print(f"After scaling : mean = {X_train_scaled.mean():.4f}, "
          f"std = {X_train_scaled.std():.4f}")

    # Step 5: Train KNN with k = 3, 5, 7
    print("\n[Step 5] Training KNN models...")

    results = {}
    for k in K_VALUES:
        model = train_knn(X_train_scaled, y_train, k)
        predictions = predict_knn(model, scaler, X_test)
        accuracy = (predictions == y_test).mean()

        results[k] = {"model": model, "predictions": predictions,
                      "accuracy": accuracy}
        print(f"  k = {k}  ->  Test Accuracy = {accuracy * 100:.2f}%")

    best_k = max(results, key=lambda k: results[k]["accuracy"])
    best = results[best_k]

    joblib.dump(best["model"], f"{OUTPUT_DIR}/knn_model.pkl")

    print(f"\nBest k = {best_k} "
          f"(Accuracy = {best['accuracy'] * 100:.2f}%)")

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating best model...")

    evaluate_model(y_test, best["predictions"], classes,
                   save_path=f"{OUTPUT_DIR}/02_confusion_matrix.png")

    matrix = confusion_matrix(y_test, best["predictions"],
                              labels=list(range(len(classes))))
    report_confusions(matrix, classes)

    # Step 7: Sweep k over a wider range
    print("\n[Step 7] Sweeping k from 1 to 31...")

    train_scores, test_scores = sweep_k(
        X_train_scaled, y_train, X_test_scaled, y_test, K_SWEEP)

    plot_k_curve(K_SWEEP, train_scores, test_scores,
                 f"{OUTPUT_DIR}/01_k_curve.png")

    best_overall = K_SWEEP[int(np.argmax(test_scores))]
    print(f"Best k over the full sweep: k = {best_overall} "
          f"({max(test_scores) * 100:.2f}%)")

    # Step 8: Save predictions
    print("\n[Step 8] Saving predictions...")

    with open(f"{OUTPUT_DIR}/predictions.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "predicted_label", "correct"])
        for i, (true, pred) in enumerate(zip(y_test, best["predictions"])):
            writer.writerow([i, classes[true], classes[pred],
                             int(true == pred)])

    print(f"Saved: {OUTPUT_DIR}/predictions.csv")

    # Summary
    print("\n" + "--" * 30)
    print("SUMMARY")
    print("--" * 30)
    print(f"Images used        : {len(images)} "
          f"({MAX_PER_CLASS} per class)")
    print(f"Image size         : {IMG_SIZE} x {IMG_SIZE} grayscale")
    print(f"Features per image : {X.shape[1]}")
    print(f"Train / Test       : {len(X_train)} / {len(X_test)}")
    for k in K_VALUES:
        mark = "  <-- best" if k == best_k else ""
        print(f"k = {k}  Accuracy = "
              f"{results[k]['accuracy'] * 100:.2f}%{mark}")
    print(f"Random guess would be {100 / len(classes):.2f}%")
    print("--" * 30)


if __name__ == "__main__":
    main()
