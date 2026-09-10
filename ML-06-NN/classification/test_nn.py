"""
test_nn.py
ทดสอบโมเดลด้วยรูปสุ่ม 4 ใบจากชุด test แล้วบันทึกเป็นภาพเปรียบเทียบ

    python test_nn.py

สีเขียว = ทายถูก   สีแดง = ทายผิด
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from config import OUTPUT_DIR, RANDOM_STATE


def run(n_samples=4):
    X_test = np.load(OUTPUT_DIR / "X_test.npy")
    y_test = np.load(OUTPUT_DIR / "y_test.npy")
    model = keras.models.load_model(OUTPUT_DIR / "nn_model.keras")

    with open(OUTPUT_DIR / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y_test), size=n_samples, replace=False)

    probs = model.predict(X_test[idx], verbose=0)
    preds = np.argmax(probs, axis=1)

    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4.5))
    for ax, i, pred, prob in zip(axes, idx, preds, probs):
        ax.imshow(X_test[i])
        ax.axis("off")
        correct = pred == y_test[i]
        # ใช้อังกฤษในภาพ เพราะฟอนต์เริ่มต้นของ matplotlib ไม่มีอักขระไทย
        ax.set_title(
            f"True: {classes[y_test[i]]}\n"
            f"Pred: {classes[pred]} ({prob[pred]:.0%})",
            color="green" if correct else "red",
            fontsize=11,
        )
        print(f"รูปที่ {i:>4}  จริง {classes[y_test[i]]:<12} "
              f"ทาย {classes[pred]:<12} ความมั่นใจ {prob[pred]:.1%} "
              f"{'ถูก' if correct else 'ผิด'}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_sample.png", dpi=140)
    plt.close()
    print(f"\nบันทึกรูปลง {OUTPUT_DIR / 'prediction_sample.png'}")


if __name__ == "__main__":
    run()
