import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_test, predictions, save_path):

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"MAE  : {mae:.2f} points")
    print(f"RMSE : {rmse:.2f} points")
    print(f"R2   : {r2:.4f}")

    plot_results(y_test, predictions, save_path)
    print(f"Saved: {save_path}")

    return mae, rmse, r2


def plot_results(y_test, predictions, save_path):

    low, high = y_test.min(), y_test.max()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(y_test, predictions, s=6, alpha=0.25)
    # A perfect model would put every point on this diagonal
    axes[0].plot([low, high], [low, high], "r--")
    axes[0].set_xlabel("Actual overall rating")
    axes[0].set_ylabel("Predicted overall rating")
    axes[0].set_title("Predicted vs Actual")

    axes[1].hist(predictions - y_test, bins=60)
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted - Actual (points)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residuals")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_samples(names, y_test, predictions, save_path, n_samples=8):

    index = np.random.choice(len(names), n_samples, replace=False)

    labels = [names[i] for i in index]
    position = np.arange(n_samples)
    height = 0.38

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(position + height / 2, y_test[index], height, label="Actual")
    ax.barh(position - height / 2, predictions[index], height,
            label="Predicted")

    ax.set_yticks(position, labels)
    ax.set_xlabel("Overall rating")
    ax.set_title("Rating prediction (green = within 3 points)")
    ax.legend()

    for tick, i in zip(ax.get_yticklabels(), index):
        close = abs(predictions[i] - y_test[i]) <= 3
        tick.set_color("green" if close else "red")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def compare_simple_multiple(results, save_path):
    """results: list of (label, r2, mae) for Simple vs Multiple."""

    print("\n------------ Simple vs Multiple ----------")
    for label, r2, mae in results:
        print(f"{label:<28} R2 = {r2:.4f}   MAE = {mae:.2f}")

    labels = [r[0] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, [r[1] for r in results], color=["#888", "#3b7dd8"])
    axes[0].set_ylabel("R2 Score")
    axes[0].set_title("R2 (higher is better)")
    axes[0].set_ylim(0, 1)

    axes[1].bar(labels, [r[2] for r in results], color=["#888", "#3b7dd8"])
    axes[1].set_ylabel("MAE (points)")
    axes[1].set_title("MAE (lower is better)")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
