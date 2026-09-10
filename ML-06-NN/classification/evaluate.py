"""
evaluate.py
ขั้นที่ 5 : วัดผลด้วยชุด test และวาดกราฟการเทรน

กราฟ training_history.png อ่านยังไง
    เส้น train กับ validation ควรลู่ลงไปด้วยกัน
    ถ้า train ยังดีขึ้นเรื่อย ๆ แต่ validation เริ่มแย่ลง = overfitting
    จุดที่ validation เริ่มแย่ลงคือจำนวน epoch ที่เหมาะสมที่สุด
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from config import OUTPUT_DIR


def plot_history(all_history):
    n = len(all_history)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)

    for col, (name, hist) in enumerate(all_history.items()):
        epochs = range(1, len(hist["accuracy"]) + 1)

        ax = axes[0][col]
        ax.plot(epochs, hist["accuracy"], label="train")
        ax.plot(epochs, hist["val_accuracy"], label="validation")
        ax.set_title(f"{name}\nAccuracy")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1][col]
        ax.plot(epochs, hist["loss"], label="train")
        ax.plot(epochs, hist["val_loss"], label="validation")
        ax.set_title("Loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history.png", dpi=140)
    plt.close()


def run():
    X_test = np.load(OUTPUT_DIR / "X_test.npy")
    y_test = np.load(OUTPUT_DIR / "y_test.npy")
    model = keras.models.load_model(OUTPUT_DIR / "nn_model.keras")

    with open(OUTPUT_DIR / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)
    with open(OUTPUT_DIR / "history.json", encoding="utf-8") as f:
        all_history = json.load(f)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print(f"\nAccuracy บนชุด test = {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=classes))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=classes, cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix - Neural Network")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    plot_history(all_history)

    print(f"บันทึกรูปลง {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"บันทึกกราฟลง {OUTPUT_DIR / 'training_history.png'}")


if __name__ == "__main__":
    run()
