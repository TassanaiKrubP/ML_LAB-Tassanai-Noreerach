import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def evaluate_model(y_test, predictions, classes, save_path):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=labels,
                                target_names=classes, zero_division=0))

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print("Confusion Matrix:")
    print(matrix)

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


def plot_roc_curve(y_test, probabilities, positive_class, save_path):

    false_positive, true_positive, _ = roc_curve(y_test, probabilities)
    area = auc(false_positive, true_positive)

    print(f"\nAUC: {area:.4f}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(false_positive, true_positive, linewidth=2,
            label=f"ROC (AUC = {area:.4f})")
    # A model that guesses at random would sit on this diagonal
    ax.plot([0, 1], [0, 1], "r--", label="Random guess")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (positive = {positive_class})")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

    return area


def plot_decision_boundary(model, X, y, feature_names, classes, save_path):
    """Only works with exactly 2 features: a page has just 2 axes."""

    padding = 5
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    # Score every point on a grid, then colour the grid by the answer
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 300),
                                 np.linspace(y_min, y_max, 300))
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    zone = model.predict(grid).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.contourf(grid_x, grid_y, zone, alpha=0.2, cmap="coolwarm")

    sample = np.random.choice(len(X), min(1500, len(X)), replace=False)
    for value, name, colour in zip([0, 1], classes, ["#3b7dd8", "#d83b3b"]):
        mask = y[sample] == value
        ax.scatter(X[sample][mask, 0], X[sample][mask, 1],
                   s=8, alpha=0.45, color=colour, label=name)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Decision Boundary (2 features)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_samples(names, positions, y_test, predictions, classes,
                 save_path, n_samples=8):

    index = np.random.choice(len(names), n_samples, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    axes = axes.ravel()

    for ax, i in zip(axes, index):
        correct = predictions[i] == y_test[i]

        ax.axis("off")
        ax.text(0.5, 0.5,
                f"{names[i]}\n({positions[i]})\n\n"
                f"Pred {classes[predictions[i]]}\n"
                f"True {classes[y_test[i]]}",
                ha="center", va="center", fontsize=10,
                color="green" if correct else "red")

    fig.suptitle("Attacker vs Midfielder prediction")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
