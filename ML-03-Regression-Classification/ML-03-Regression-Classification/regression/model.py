from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X_train, y_train, pca_components=15, alpha=1.0):
    # One pipeline, so the test set always gets the same transform
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components, random_state=42),
        Ridge(alpha=alpha),
    )

    model.fit(X_train, y_train)

    return model


def train_simple_model(X_train, y_train):
    """Simple Linear Regression: one feature, no PCA.

    Kept next to the full model so Simple vs Multiple can be compared
    on exactly the same train/test split.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def predict_model(model, X_test):
    # Ridge can predict outside the real rating range, so clip it
    return model.predict(X_test).clip(1, 99)
