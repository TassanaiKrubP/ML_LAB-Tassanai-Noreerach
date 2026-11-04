"""
visualize_features.py
ดูว่า CNN "มองเห็น" อะไรในแต่ละชั้น (Feature Map แบบในสไลด์ Layers A / B / C)

    feature_maps.png          ผลลัพธ์หลัง Conv + ReLU ของชั้นที่ 1, 2, 3 สำหรับรูป test 1 ใบ
                              ชั้นแรกจะเห็นขอบ/สี  ชั้นลึกขึ้นจะเล็กลงและเป็นลวดลายนามธรรมมากขึ้น
    first_layer_kernels.png   น้ำหนัก filter 3x3 ของชั้น conv แรก (filter ที่โมเดลเรียนมาเอง)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras

from config import OUTPUT_DIR, EXPERIMENT_DIR
from split_data import load_splits

N_MAPS = 12   # จำนวน feature map ที่แสดงต่อชั้น


def plot_feature_maps(model, image, true_name, out_path):
    conv_names = [l.name for l in model.layers if isinstance(l, keras.layers.Conv2D)][:3]
    probe = keras.Model(model.inputs, [model.get_layer(n).output for n in conv_names])
    maps = probe.predict(image[None].astype("float32"), verbose=0)
    if len(conv_names) == 1:
        maps = [maps]

    rows = len(conv_names)
    fig, axes = plt.subplots(rows, N_MAPS + 1, figsize=(1.45 * (N_MAPS + 1), 1.65 * rows + 0.6))
    axes = np.atleast_2d(axes)
    for r, (name, fmap) in enumerate(zip(conv_names, maps)):
        fmap = fmap[0]
        axes[r, 0].imshow(image)
        axes[r, 0].set_title(f"{name}\n{fmap.shape[0]}x{fmap.shape[1]}x{fmap.shape[2]}", fontsize=8)
        # เลือก filter ที่ตอบสนองแรงที่สุด จะเห็นภาพชัดกว่าเลือกตามลำดับ
        top = np.argsort(fmap.mean(axis=(0, 1)))[::-1][:N_MAPS]
        for c, ch in enumerate(top):
            axes[r, c + 1].imshow(fmap[:, :, ch], cmap="viridis")
            axes[r, c + 1].set_title(f"#{ch}", fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Feature maps after Conv + ReLU (test image, true class: {true_name})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_first_kernels(model, out_path):
    first = next(l for l in model.layers if isinstance(l, keras.layers.Conv2D))
    w = first.get_weights()[0]                  # (k, k, 3, n_filters)
    n = w.shape[-1]
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            k = w[:, :, :, i]
            k = (k - k.min()) / (k.max() - k.min() + 1e-8)   # ยืดค่าเป็น 0-1 เพื่อแสดงเป็นสี RGB
            ax.imshow(k, interpolation="nearest")
            ax.set_title(f"#{i}", fontsize=7)
    fig.suptitle(f"Learned {w.shape[0]}x{w.shape[1]} kernels of the first conv layer ({n} filters)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(run_name):
    model = keras.models.load_model(os.path.join(EXPERIMENT_DIR, run_name, "model.keras"))
    with open(os.path.join(OUTPUT_DIR, "classes.json"), encoding="utf-8") as f:
        classes = json.load(f)
    _, _, _, _, X_test, y_test = load_splits()

    plot_feature_maps(model, X_test[0], classes[y_test[0]], os.path.join(OUTPUT_DIR, "feature_maps.png"))
    plot_first_kernels(model, os.path.join(OUTPUT_DIR, "first_layer_kernels.png"))
    print("  บันทึก feature_maps.png และ first_layer_kernels.png")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "conv3")
