"""
test_vgg.py
สุ่มรูปจากชุด test 4 ใบ ให้โมเดลที่ดีที่สุดทาย  หัวข้อสีเขียว = ถูก  สีแดง = ผิด

    python test_vgg.py                    ใช้โมเดลที่ดีที่สุด (outputs/best_model.json)
    python test_vgg.py vgg16_transfer     เลือกโมเดลเอง
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras

from config import OUTPUT_DIR, EXPERIMENT_DIR, RANDOM_STATE
from split_data import load_splits


def prepare(images, run_name):
    """โมเดล transfer / finetune ต้องใช้ preprocess ของ VGG16 แบบเดียวกับตอนเทรน"""
    with open(os.path.join(EXPERIMENT_DIR, run_name, "result.json"), encoding="utf-8") as f:
        kind = json.load(f)["kind"]
    x = images.astype("float32")
    if kind in ("transfer", "finetune"):
        x = keras.applications.vgg16.preprocess_input(x)
    return x


def run(run_name=None):
    if run_name is None:
        with open(os.path.join(OUTPUT_DIR, "best_model.json"), encoding="utf-8") as f:
            run_name = json.load(f)["name"]
    model = keras.models.load_model(os.path.join(EXPERIMENT_DIR, run_name, "model.keras"))
    with open(os.path.join(OUTPUT_DIR, "classes.json"), encoding="utf-8") as f:
        classes = json.load(f)
    _, _, _, _, X_test, y_test = load_splits()

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test), 4, replace=False)
    probs = model.predict(prepare(X_test[idx], run_name), verbose=0)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, i, p in zip(axes, idx, probs):
        pred, true = int(np.argmax(p)), int(y_test[i])
        ax.imshow(X_test[i])
        ax.set_title(f"pred: {classes[pred]} ({p[pred]:.0%})\ntrue: {classes[true]}",
                     color="#0ca30c" if pred == true else "#e34948", fontsize=10)
        ax.axis("off")
        print(f"  รูป {i:>4}  ทาย {classes[pred]:<12} ({p[pred]:.0%})  จริง {classes[true]}")
    fig.suptitle(f"Predictions on random test images  (model: {run_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "prediction_sample.png"), dpi=120)
    plt.close(fig)
    print("  บันทึก prediction_sample.png")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
