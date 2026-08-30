import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def fit_cluster_knn(X, clusters, k=5):
    """Train a KNN that reproduces the cluster assignment.

    KMeans can only label the data it was fitted on. Training a KNN on the
    cluster ids turns that one-off grouping into something reusable: a new
    image can be dropped into the nearest existing cluster without refitting
    KMeans. This is also the link between the unsupervised half of the lab
    and the supervised half.
    """

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, clusters)

    return model


def cluster_purity(clusters, labels, n_clusters, n_classes):
    """How cleanly each cluster maps onto one true class.

    For every cluster, take the true class that appears most often in it and
    divide by the size of the cluster. A value near 1.0 means the cluster is
    almost entirely one class; a value near 1/n_classes means the cluster is
    a random mix and the grouping found nothing useful.
    """

    purities = []

    for cluster_id in range(n_clusters):
        members = labels[clusters == cluster_id]
        if len(members) == 0:
            purities.append(0.0)
            continue

        counts = np.bincount(members, minlength=n_classes)
        purities.append(counts.max() / len(members))

    return purities


def cluster_class_table(clusters, labels, n_clusters, n_classes):
    """Cross-tabulate clusters against true classes."""

    table = np.zeros((n_classes, n_clusters), dtype=int)

    for class_id in range(n_classes):
        member_clusters = clusters[labels == class_id]
        for cluster_id in range(n_clusters):
            table[class_id, cluster_id] = (member_clusters == cluster_id).sum()

    return table
