from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X_train, y_train, pca_components=10, C=1.0):
    # One pipeline, so the test set always gets the same transform.
    # PCA squeezes 28 correlated skills into a few independent axes.
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components, random_state=42),
        LogisticRegression(C=C, max_iter=1000),
    )

    model.fit(X_train, y_train)

    return model


def train_2d_model(X_train, y_train, C=1.0):
    """Two features only, no PCA, so the boundary can be drawn."""

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=1000),
    )

    model.fit(X_train, y_train)

    return model


def predict_model(model, X_test):

    return model.predict(X_test)


def predict_proba_model(model, X_test):
    # Probability of the positive class, needed for the ROC curve
    return model.predict_proba(X_test)[:, 1]
