from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def build_scaler(X_train):
    """Fit a StandardScaler on the training set only.

    KNN decides by distance, and distance is the sum of the differences of
    every feature. A feature on a larger numeric scale would dominate that
    sum and drown out the rest, so all features must be put on the same
    scale first. The scaler is fitted on X_train alone: fitting it on the
    test set as well would leak information the model is not allowed to see.
    """

    scaler = StandardScaler()
    scaler.fit(X_train)

    return scaler


def train_knn(X_train, y_train, k=5):
    """Train one KNN classifier.

    Nothing is really learned here. KNN is a lazy learner: fit() only stores
    the training set, and the real work happens at predict() time when the
    distance to every stored sample is computed.
    """

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    return model


def predict_knn(model, scaler, X_test):
    # Apply the same scaling used for training data
    X_test_scaled = scaler.transform(X_test)
    # Predict
    predictions = model.predict(X_test_scaled)

    return predictions


def sweep_k(X_train_scaled, y_train, X_test_scaled, y_test, k_values):
    """Train one model per k and record train/test accuracy for each."""

    train_scores = []
    test_scores = []

    for k in k_values:
        model = train_knn(X_train_scaled, y_train, k)
        train_scores.append(model.score(X_train_scaled, y_train))
        test_scores.append(model.score(X_test_scaled, y_test))

    return train_scores, test_scores
