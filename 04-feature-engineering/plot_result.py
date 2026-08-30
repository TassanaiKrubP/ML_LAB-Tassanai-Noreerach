import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_encoding_comparison(df, onehot, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ซ้าย: label encoding เห็นเป็นแท่งเลขเดียว
    counts = df.groupby("team")["team_label"].first().sort_values()
    axes[0].barh(counts.index, counts.values, color="steelblue")
    axes[0].set_xlabel("team_label")
    axes[0].set_title("Label Encoding\n(1 column, but implies a fake order)")

    # ขวา: one-hot เห็นเป็นตารางศูนย์กับหนึ่ง
    sample = onehot.head(12)
    axes[1].imshow(sample.values, cmap="Blues", aspect="auto")
    axes[1].set_xticks(range(sample.shape[1]))
    axes[1].set_xticklabels(
        [c.replace("team_", "") for c in sample.columns], rotation=45, ha="right"
    )
    axes[1].set_ylabel("row index")
    axes[1].set_title(f"One-Hot Encoding\n({onehot.shape[1]} columns, no fake order)")

    save(fig, output_dir, "encoding_comparison.png")


def plot_scaling_comparison(original, standard, minmax, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    datasets = [
        (original, "Original", "grey"),
        (standard, "StandardScaler", "steelblue"),
        (minmax, "MinMaxScaler", "seagreen"),
    ]

    for ax, (data, title, color) in zip(axes, datasets):
        ax.boxplot([data[column] for column in data.columns], tick_labels=data.columns)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Feature scaling comparison")
    save(fig, output_dir, "scaling_comparison.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
