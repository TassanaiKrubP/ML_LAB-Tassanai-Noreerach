import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_histograms(numeric, output_dir):
    columns = list(numeric.columns)
    rows = (len(columns) + 2) // 3

    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.2 * rows))
    axes = axes.ravel()

    for ax, column in zip(axes, columns):
        ax.hist(numeric[column].dropna(), bins=20, color="steelblue", edgecolor="white")
        ax.set_title(column)

    # ปิดช่องที่เหลือไม่ให้เป็นกรอบเปล่า
    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle("Histogram of numeric columns (raw data)")
    save(fig, output_dir, "histograms.png")


def plot_correlation_heatmap(corr, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    # เขียนตัวเลขลงในแต่ละช่อง
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax)
    ax.set_title("Correlation heatmap")

    save(fig, output_dir, "correlation_heatmap.png")


def plot_boxplot(df, output_dir):
    columns = ["age", "height_cm", "career_wins"]
    data = [df[column].dropna() for column in columns]

    fig, axes = plt.subplots(1, len(columns), figsize=(11, 4))

    for ax, values, column in zip(axes, data, columns):
        ax.boxplot(values, vert=True)
        ax.set_title(column)

    fig.suptitle("Boxplot - looking for outliers")
    save(fig, output_dir, "boxplot_outliers.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
