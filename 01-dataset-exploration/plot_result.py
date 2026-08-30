import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_class_distribution(df, output_dir):
    counts = df["team"].value_counts(dropna=False)

    # แปลง NaN ให้เป็นข้อความ ไม่งั้น matplotlib วาดแกนไม่ได้
    labels = [str(name) if isinstance(name, str) else "(missing)" for name in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, counts.values, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Number of drivers")
    ax.set_title("Class distribution of team (before cleaning)")

    save(fig, output_dir, "class_distribution.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
