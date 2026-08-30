import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)
    print("Confusion Matrix:")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_k_curve(k_values, train_scores, test_scores, save_path):
    """Train vs test accuracy across k.

    At k = 1 the training accuracy is always 100%, because the nearest
    neighbour of a training sample is the sample itself. The gap between the
    two lines at that point is what overfitting looks like.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_values, train_scores, "o-", label="Train accuracy")
    ax.plot(k_values, test_scores, "s-", label="Test accuracy")

    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Effect of k on KNN performance")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def report_confusions(matrix, classes):
    """List which class pairs the model mixes up most."""

    errors = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and matrix[i, j] > 0:
                errors.append((matrix[i, j], classes[i], classes[j]))

    errors.sort(reverse=True)

    print("\nMost confused class pairs:")
    for count, true_class, pred_class in errors[:5]:
        print(f"  True {true_class:<12} -> Predicted {pred_class:<12} "
              f": {count} images")
