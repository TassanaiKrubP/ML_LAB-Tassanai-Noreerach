from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_transformer(X, n_components=50):
    """Standardize, then reduce dimensions with PCA.

    KMeans groups points by distance, exactly like KNN, so the same two
    problems apply: features must share a scale, and too many dimensions
    make all distances look alike. PCA compresses 1024 pixel features into
    a few dozen components that keep most of the variance.
    """

    transformer = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(n_components, *X.shape),
                    random_state=42)),
    ])

    transformer.fit(X)

    return transformer


def find_elbow(X, k_range):
    """Run KMeans for each k and collect the inertia.

    Inertia is the total squared distance from every point to the centre of
    its own cluster, so it always falls as k rises. The useful k is at the
    "elbow", where adding another cluster stops buying much improvement.
    """

    inertias = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)
        print(f"  k = {k:>2}  inertia = {model.inertia_:.2f}")

    return inertias


def train_kmeans(X, n_clusters=3):
    """Fit the final KMeans model and return the cluster of each sample."""

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X)

    return model, clusters
