import csv
import json
import os

import joblib
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from kmeans_model import build_transformer, find_elbow, train_kmeans
from knn_tools import fit_cluster_knn, cluster_purity, cluster_class_table
from visualize import plot_elbow, plot_clusters

# Paths are resolved from this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = "outputs"

IMG_SIZE = 32
MAX_PER_CLASS = 500
N_CLUSTERS = 3           # matches the number of true classes
K_RANGE = range(1, 11)   # for the elbow plot
PCA_COMPONENTS = 50


def main():

    print("--" * 30)
    print("KMeans Clustering: Aeroplanes vs Birds vs Drones")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes, paths = load_data(
        DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print(f"\nTotal images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    # The true labels are loaded but never given to KMeans. They are only
    # used at the end to check how good the discovered grouping turned out.
    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)
    print(f"Feature shape: {X.shape}")

    # Step 3: Feature Scaling and dimensionality reduction
    print("\n[Step 3] Standardizing and reducing dimensions...")

    transformer = build_transformer(X, PCA_COMPONENTS)
    X_reduced = transformer.transform(X)

    joblib.dump(transformer, f"{OUTPUT_DIR}/transformer.pkl")

    explained = transformer.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"Reduced {X.shape[1]} features to {X_reduced.shape[1]} components")
    print(f"Variance retained: {explained * 100:.2f}%")

    # Step 4: Elbow method
    print("\n[Step 4] Running elbow method...")

    inertias = find_elbow(X_reduced, K_RANGE)
    plot_elbow(K_RANGE, inertias, f"{OUTPUT_DIR}/01_elbow.png")

    # Step 5: Final clustering
    print(f"\n[Step 5] Clustering into {N_CLUSTERS} groups...")

    model, clusters = train_kmeans(X_reduced, N_CLUSTERS)
    joblib.dump(model, f"{OUTPUT_DIR}/kmeans_model.pkl")

    plot_clusters(X_reduced, clusters, labels, classes,
                  f"{OUTPUT_DIR}/02_clusters.png")

    # Step 6: Compare clusters against the true classes
    print("\n[Step 6] Comparing clusters with true classes...")

    table = cluster_class_table(clusters, labels, N_CLUSTERS, len(classes))

    header = f"{'':<14}" + "".join(
        f"{'Cluster ' + str(i):>12}" for i in range(N_CLUSTERS))
    print(header)
    print("-" * len(header))
    for class_id, class_name in enumerate(classes):
        row = "".join(f"{v:>12}" for v in table[class_id])
        print(f"{class_name:<14}{row}")

    purities = cluster_purity(clusters, labels, N_CLUSTERS, len(classes))
    print("\nCluster purity (1.00 = one class only, "
          f"{1 / len(classes):.2f} = random mix):")
    for cluster_id, purity in enumerate(purities):
        size = int((clusters == cluster_id).sum())
        print(f"  Cluster {cluster_id}: {purity:.3f}  ({size} images)")

    with open(f"{OUTPUT_DIR}/cluster_summary.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class"] +
                        [f"cluster_{i}" for i in range(N_CLUSTERS)])
        for class_id, class_name in enumerate(classes):
            writer.writerow([class_name] + list(table[class_id]))
        writer.writerow([])
        writer.writerow(["cluster", "size", "purity"])
        for cluster_id, purity in enumerate(purities):
            writer.writerow([cluster_id,
                             int((clusters == cluster_id).sum()),
                             round(purity, 4)])

    print(f"Saved: {OUTPUT_DIR}/cluster_summary.csv")

    # Step 7: Turn the clustering into something reusable with KNN
    print("\n[Step 7] Fitting KNN on the cluster assignment...")

    cluster_knn = fit_cluster_knn(X_reduced, clusters, k=5)
    joblib.dump(cluster_knn, f"{OUTPUT_DIR}/cluster_knn.pkl")

    agreement = (cluster_knn.predict(X_reduced) == clusters).mean()
    print(f"KNN reproduces the KMeans grouping {agreement * 100:.2f}% "
          "of the time")

    # Step 8: Save per-image results
    print("\n[Step 8] Saving clustered image list...")

    with open(f"{OUTPUT_DIR}/clustered_images.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "true_class", "cluster"])
        for path, label, cluster_id in zip(paths, labels, clusters):
            writer.writerow([path, classes[label], int(cluster_id)])

    print(f"Saved: {OUTPUT_DIR}/clustered_images.csv")
    print("\n" + "--" * 30)


if __name__ == "__main__":
    main()
