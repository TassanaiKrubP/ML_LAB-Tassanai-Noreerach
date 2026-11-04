"""
evaluate.py
ขั้นที่ 6 : วัดผลโมเดล  accuracy, classification report, confusion matrix, กราฟการเทรน
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def plot_history(history, path, title):
    """กราฟ train/val accuracy และ loss ต่อ epoch  ถ้าเส้น val loss ขึ้นขณะ train loss ลง = overfitting"""
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["accuracy"], color="#1baf7a", lw=2, label="train")
    axes[0].plot(epochs, history["val_accuracy"], color="#eb6834", lw=2, label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["loss"], color="#1baf7a", lw=2, label="train")
    axes[1].plot(epochs, history["val_loss"], color="#eb6834", lw=2, label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def predict_labels(model, ds):
    probs = model.predict(ds, verbose=0)
    return np.argmax(probs, axis=1)


def evaluate_model(model, test_ds, y_test, classes, out_dir, title="Best model"):
    """บันทึก classification_report.txt และ confusion_matrix.png แล้วคืนค่า accuracy"""
    y_pred = predict_labels(model, test_ds)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)

    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"{title}\nTest accuracy: {acc:.4f}\n\n{report}")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title}  (test acc {acc:.2%})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)
    return acc
