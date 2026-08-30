import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_before_after(original, cleaned, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(original["age"].dropna(), bins=30, color="tomato", edgecolor="white")
    axes[0].set_title(f"Before cleaning (n = {original['age'].notna().sum()})")
    axes[0].set_xlabel("age")

    axes[1].hist(cleaned["age"], bins=30, color="seagreen", edgecolor="white")
    axes[1].set_title(f"After cleaning (n = {len(cleaned)})")
    axes[1].set_xlabel("age")

    fig.suptitle("Age distribution before and after cleaning")
    save(fig, output_dir, "before_after_cleaning.png")


def plot_mean_vs_median(raw, filled_mean, filled_median, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    datasets = [
        (raw.dropna(), "Original (with NaN)", "grey"),
        (filled_mean, "fillna(mean)", "steelblue"),
        (filled_median, "fillna(median)", "seagreen"),
    ]

    for ax, (values, title, color) in zip(axes, datasets):
        ax.hist(values, bins=25, color=color, edgecolor="white")
        ax.set_title(f"{title}\nstd = {values.std():.2f}")
        ax.set_xlabel("age")

    fig.suptitle("Comparing mean and median imputation")
    save(fig, output_dir, "mean_vs_median.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
