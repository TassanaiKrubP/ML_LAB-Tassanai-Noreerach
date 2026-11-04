"""
04-evaluate/plot_result.py
วาดกราฟจาก outputs/metrics.csv และ results.csv ของแต่ละการทดลอง
    compare_epochs.png        P / R / mAP50 / mAP50-95 บนชุด test ของ 10 / 30 / 60 epoch
    compare_imgsz_batch.png   ขนาดรูป 320 / 640  x  batch 8 / 16
    metrics_vs_epochs.png     val mAP ทุก epoch ของการเทรน 60 epoch
    loss_curves.png           train vs val loss (box, cls, dfl) ของการทดลองที่ดีที่สุด
"""

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EVAL_DIR, BEST_RUN_FILE, TRAIN_DIR

METRICS = ["precision", "recall", "mAP50", "mAP50-95"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # สีคงที่ต่อ metric ทุกกราฟ
C_TRAIN, C_VAL, C_GRAY = "#1baf7a", "#eb6834", "#8a8984"


def style(ax, title, ylabel=None):
    ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def load_metrics():
    with open(os.path.join(EVAL_DIR, "metrics.csv"), encoding="utf-8") as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def read_history(run):
    with open(os.path.join(TRAIN_DIR, run, "results.csv"), encoding="utf-8") as f:
        return [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def grouped_metrics(ax, rows, labels):
    n = len(METRICS)
    w = 0.8 / n
    for j, (metric, color) in enumerate(zip(METRICS, COLORS)):
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(rows))]
        vals = [float(r[metric]) for r in rows]
        bars = ax.bar(xs, vals, w - 0.02, color=color, label=metric)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="upper left")


def compare(rows_all, names, labels, title, out_name):
    rows = [rows_all[n] for n in names if n in rows_all]
    labels = [l for n, l in zip(names, labels) if n in rows_all]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    grouped_metrics(axes[0], rows, labels)
    style(axes[0], "Test-set metrics", "score")
    mins = [float(r["train_time_min"]) for r in rows]
    bars = axes[1].bar(range(len(rows)), mins, 0.55, color=C_GRAY)
    axes[1].bar_label(bars, labels=[f"{m:.1f} min" for m in mins], fontsize=8, padding=2)
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels(labels, fontsize=8.5)
    style(axes[1], "Training time (RTX 3050)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, out_name), dpi=120)
    plt.close(fig)
    print(f"  บันทึก {out_name}")


def metrics_vs_epochs(rows_all, run="e60"):
    if not os.path.exists(os.path.join(TRAIN_DIR, run, "results.csv")):
        return
    hist = read_history(run)
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(ep, [h["metrics/mAP50(B)"] for h in hist], color=COLORS[2], lw=2, label="val mAP50 (e60 run)")
    ax.plot(ep, [h["metrics/mAP50-95(B)"] for h in hist], color=COLORS[3], lw=2, label="val mAP50-95 (e60 run)")
    # จุด = ผลบนชุด test ของการทดลองที่เทรน 10 / 30 / 60 epoch แยกกัน
    for name, n in [("e10", 10), ("e30", 30), ("e60", 60)]:
        if name in rows_all:
            ax.plot(n, float(rows_all[name]["mAP50"]), "o", ms=9, color=COLORS[2], mec="white", mew=2)
            ax.plot(n, float(rows_all[name]["mAP50-95"]), "o", ms=9, color=COLORS[3], mec="white", mew=2)
    ax.plot([], [], "o", color=C_GRAY, label="test score of separate e10 / e30 / e60 runs")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, loc="lower right")
    style(ax, "Detection performance vs epochs", "mAP")
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "metrics_vs_epochs.png"), dpi=120)
    plt.close(fig)
    print("  บันทึก metrics_vs_epochs.png")


def loss_curves(run):
    hist = read_history(run)
    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, part, desc in zip(axes, ["box", "cls", "dfl"],
                              ["box loss (where is the box)", "cls loss (which class)",
                               "dfl loss (box edge refinement)"]):
        ax.plot(ep, [h[f"train/{part}_loss"] for h in hist], color=C_TRAIN, lw=2, label="train")
        ax.plot(ep, [h[f"val/{part}_loss"] for h in hist], color=C_VAL, lw=2, label="validation")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8.5)
        style(ax, desc)
    fig.suptitle(f"Training vs validation loss  (best run: {run})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "loss_curves.png"), dpi=120)
    plt.close(fig)
    print("  บันทึก loss_curves.png")


def main():
    rows = load_metrics()
    compare(rows, ["e10", "e30", "e60"], ["10 epochs", "30 epochs", "60 epochs"],
            "Number of epochs (imgsz 640, batch 16)", "compare_epochs.png")
    compare(rows, ["s320_b8", "s320_b16", "s640_b8", "s640_b16"],
            ["320 / b8", "320 / b16", "640 / b8", "640 / b16"],
            "Image size / batch size (30 epochs)", "compare_imgsz_batch.png")
    metrics_vs_epochs(rows)
    with open(BEST_RUN_FILE, encoding="utf-8") as f:
        loss_curves(json.load(f)["run"])


if __name__ == "__main__":
    main()
