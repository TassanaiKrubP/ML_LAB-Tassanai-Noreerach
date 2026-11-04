"""
plot_result.py
ขั้นที่ 7 : วาดกราฟเปรียบเทียบผลการทดลองจาก outputs/experiments.csv
    compare_conv_layers.png   จำนวนชั้น conv  vs  accuracy / พารามิเตอร์ / เวลา
    compare_neurons.png       จำนวนนิวรอนใน Dense  vs  accuracy / พารามิเตอร์ / เวลา
    compare_kernel.png        kernel 3x3 vs 5x5 (โบนัส)
    accuracy_vs_epochs.png    test accuracy ที่ epoch 5/10/15/20/30 + เส้น train/val
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, EXPERIMENT_DIR

# สีชุดเดียวกันทุกกราฟ : test = น้ำเงิน, validation = ส้ม, train = เขียวอมฟ้า
C_TEST, C_VAL, C_TRAIN, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def load_rows():
    with open(os.path.join(OUTPUT_DIR, "experiments.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["name"]: r for r in rows}


def style(ax, title, ylabel=None):
    ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def compare_bars(rows, names, labels, title, out_name):
    """3 กราฟย่อยแยกกัน (ไม่ใช้แกน y สองข้าง) : accuracy, จำนวนพารามิเตอร์, เวลาเทรน"""
    pairs = [(rows[n], l) for n, l in zip(names, labels) if n in rows]
    if not pairs:
        return
    rows, labels = [p[0] for p in pairs], [p[1] for p in pairs]
    x = range(len(rows))
    test = [float(r["test_accuracy"]) for r in rows]
    val = [float(r["best_val_accuracy"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    w = 0.38
    b1 = axes[0].bar([i - w / 2 - 0.01 for i in x], test, w, color=C_TEST, label="test accuracy")
    axes[0].bar([i + w / 2 + 0.01 for i in x], val, w, color=C_VAL, label="best val accuracy")
    axes[0].bar_label(b1, labels=[f"{v:.1%}" for v in test], fontsize=8, padding=2)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=8)
    style(axes[0], "Accuracy", "accuracy")

    params = [int(r["params"]) for r in rows]
    b2 = axes[1].bar(x, params, 0.55, color=C_GRAY)
    axes[1].bar_label(b2, labels=[f"{p/1e3:.0f}k" for p in params], fontsize=8, padding=2)
    style(axes[1], "Trainable + non-trainable parameters")

    secs = [float(r["train_time_s"]) for r in rows]
    b3 = axes[2].bar(x, secs, 0.55, color=C_GRAY)
    axes[2].bar_label(b3, labels=[f"{s:.0f}s" for s in secs], fontsize=8, padding=2)
    style(axes[2], "Training time (seconds, CPU)")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=120)
    plt.close(fig)
    print(f"  บันทึก {out_name}")


def accuracy_vs_epochs(run_name="epochs_30"):
    run_dir = os.path.join(EXPERIMENT_DIR, run_name)
    if not os.path.exists(os.path.join(run_dir, "result.json")):
        return
    with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as f:
        result = json.load(f)
    with open(os.path.join(run_dir, "history.json"), encoding="utf-8") as f:
        hist = json.load(f)

    cps = sorted((int(k), v) for k, v in result["checkpoints"].items())
    eps = [e for e, _ in cps]
    test = [v["test_accuracy"] for _, v in cps]
    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(eps, test, "o-", color=C_TEST, lw=2, ms=8, label="test accuracy")
    for e, a in zip(eps, test):
        axes[0].annotate(f"{a:.1%}", (e, a), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
    axes[0].set_xticks(eps)
    axes[0].set_xlabel("epochs trained")
    axes[0].set_ylim(min(test) - 0.08, min(1.0, max(test) + 0.06))
    style(axes[0], "Test accuracy vs number of epochs", "accuracy")

    axes[1].plot(epochs, hist["accuracy"], color=C_TRAIN, lw=2, label="train")
    axes[1].plot(epochs, hist["val_accuracy"], color=C_VAL, lw=2, label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=9)
    style(axes[1], "Train vs validation accuracy (same run)", "accuracy")

    fig.suptitle("Baseline CNN (3 conv blocks 32-64-128, dense 128) trained 30 epochs, no early stopping",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_epochs.png"), dpi=120)
    plt.close(fig)
    print("  บันทึก accuracy_vs_epochs.png")


def run():
    rows = load_rows()
    compare_bars(rows, ["conv1", "conv2", "conv3", "conv4"],
                 ["1 conv\n32", "2 conv\n32-64", "3 conv\n32-64-128", "4 conv\n32-...-256"],
                 "Number of convolutional layers (dense 128, kernel 3x3)", "compare_conv_layers.png")
    compare_bars(rows, ["dense64", "conv3", "dense256"],
                 ["dense 64", "dense 128", "dense 256"],
                 "Number of neurons in the dense layer (3 conv blocks)", "compare_neurons.png")
    compare_bars(rows, ["conv3", "kernel5"], ["kernel 3x3", "kernel 5x5"],
                 "Kernel size (3 conv blocks, dense 128)", "compare_kernel.png")
    accuracy_vs_epochs()


if __name__ == "__main__":
    run()
