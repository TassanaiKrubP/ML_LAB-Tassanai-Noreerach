import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_elbow(k_range, inertias, save_path):

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(list(k_range), inertias, "o-")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow Method")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_clusters(X, clusters, labels, classes, save_path):
    """Same data twice: coloured by KMeans cluster, and by true class.

    If the two panels show a similar arrangement, the natural grouping of the
    pixels matches the classes we care about. If they look unrelated, the raw
    pixel values do not carry the information needed to separate the classes.
    """

    # Project onto two components so the data can be drawn on a flat plot
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.6,
                        label=f"Cluster {cluster_id}")

    axes[0].set_title("Coloured by KMeans cluster (unsupervised)")

    for class_id, class_name in enumerate(classes):
        mask = labels == class_id
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.6,
                        label=class_name)

    axes[1].set_title("Coloured by true class (ground truth)")

    for ax in axes:
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
