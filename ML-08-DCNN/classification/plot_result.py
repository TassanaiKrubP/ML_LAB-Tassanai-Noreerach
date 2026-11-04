"""
plot_result.py
ขั้นที่ 7 : วาดกราฟเปรียบเทียบผลการทดลองจาก outputs/experiments.csv
    compare_architectures.png   accuracy / พารามิเตอร์ / เวลาเทรน ของแต่ละสถาปัตยกรรม
    compare_neurons.png         จำนวนนิวรอนในหัว Dense
    accuracy_vs_epochs.png      test accuracy ที่ epoch 5/10/15/20/30 + เส้น train/val
    all_labs_comparison.png     KNN / SVM / NN / CNN (Lab 7) / DCNN (Lab 8)
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, EXPERIMENT_DIR, LAB7_OUTPUT_DIR, PREVIOUS_LABS
from experiments import load_choices

C_TEST, C_VAL, C_TRAIN, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def load_rows():
    with open(os.path.join(OUTPUT_DIR, "experiments.csv"), encoding="utf-8") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    w = 0.38
    b1 = axes[0].bar([i - w / 2 - 0.01 for i in x], test, w, color=C_TEST, label="test accuracy")
    axes[0].bar([i + w / 2 + 0.01 for i in x], val, w, color=C_VAL, label="best val accuracy")
    axes[0].bar_label(b1, labels=[f"{v:.1%}" for v in test], fontsize=8, padding=2)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=8)
    style(axes[0], "Accuracy", "accuracy")

    # แท่งเต็ม = พารามิเตอร์ทั้งหมด  แท่งเข้ม = ส่วนที่เทรนได้ (transfer แช่แข็ง conv ไว้)
    total = [int(r["params"]) / 1e6 for r in rows]
    trainable = [int(r["trainable_params"]) / 1e6 for r in rows]
    axes[1].bar(x, total, 0.55, color="#d7d6d1", label="total")
    b2 = axes[1].bar(x, trainable, 0.55, color=C_GRAY, label="trainable")
    axes[1].bar_label(b2, labels=[f"{t:.2f}M / {a:.2f}M" for t, a in zip(trainable, total)],
                      fontsize=7, padding=2)
    axes[1].legend(fontsize=8)
    style(axes[1], "Parameters (millions): trainable / total")

    secs = [float(r["train_time_s"]) / 60 for r in rows]
    b3 = axes[2].bar(x, secs, 0.55, color=C_GRAY)
    axes[2].bar_label(b3, labels=[f"{s:.1f} min" for s in secs], fontsize=8, padding=2)
    style(axes[2], "Training time (minutes, CPU)")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8.5)
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
    axes[0].plot(eps, test, "o-", color=C_TEST, lw=2, ms=8)
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

    fig.suptitle(f"{result['description']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_epochs.png"), dpi=120)
    plt.close(fig)
    print("  บันทึก accuracy_vs_epochs.png")


def all_labs_comparison(dcnn_best):
    scores = dict(PREVIOUS_LABS)
    lab7_best = os.path.join(LAB7_OUTPUT_DIR, "best_model.json")
    if os.path.exists(lab7_best):
        with open(lab7_best, encoding="utf-8") as f:
            r = json.load(f)
        scores[f"CNN (Lab 7)\n{r['name']}"] = float(r["test_accuracy"])
    scores[f"DCNN (Lab 8)\n{dcnn_best['name']}"] = float(dcnn_best["test_accuracy"])
    # แสดง transfer + fine-tune ด้วย ถ้าไม่ใช่ตัวที่ถูกเลือก (เลือกจาก val ซึ่งมีแค่ 137 รูป ต่างกัน 1-2 รูปก็สลับอันดับได้)
    ft = os.path.join(EXPERIMENT_DIR, "vgg16_finetune", "result.json")
    if dcnn_best["name"] != "vgg16_finetune" and os.path.exists(ft):
        with open(ft, encoding="utf-8") as f:
            scores["DCNN (Lab 8)\nvgg16_finetune"] = float(json.load(f)["test_accuracy"])

    names, vals = list(scores), list(scores.values())
    colors = [C_TEST if n.startswith("DCNN") else C_GRAY for n in names]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bars = ax.bar(range(len(names)), vals, 0.6, color=colors)
    ax.bar_label(bars, labels=[f"{v:.2%}" for v in vals], fontsize=9, padding=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.05)
    style(ax, "Test accuracy across labs (Airplane / Bird / Drone)", "test accuracy")
    fig.text(0.01, 0.01, "Lab 4-6 used a different split without duplicate removal; "
             "Lab 7 and Lab 8 share the same deduplicated test set.", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUTPUT_DIR, "all_labs_comparison.png"), dpi=120)
    plt.close(fig)
    print("  บันทึก all_labs_comparison.png")


def run(best_row=None):
    rows = load_rows()
    compare_bars(rows, ["vgg_small", "vgg16_scratch", "vgg16_transfer", "vgg16_finetune"],
                 ["VGG_SMALL\nscratch (10 conv)", "VGG16\nscratch (13 conv)",
                  "VGG16 transfer\n(frozen)", "VGG16 transfer\n+ fine-tune"],
                 "DCNN architectures", "compare_architectures.png")
    base = load_choices().get("best_scratch")
    if base:
        compare_bars(rows, [base, "neurons_256_128", "neurons_128_64"],
                     ["dense 512-256", "dense 256-128", "dense 128-64"],
                     f"Neurons in the dense head ({base}, from scratch)", "compare_neurons.png")
    accuracy_vs_epochs()
    if best_row:
        all_labs_comparison(best_row)


if __name__ == "__main__":
    best_file = os.path.join(OUTPUT_DIR, "best_model.json")
    best = json.load(open(best_file, encoding="utf-8")) if os.path.exists(best_file) else None
    run(best)
