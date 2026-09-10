"""
evaluate.py
ขั้นที่ 5 : วัดผลโมเดลที่ดีที่สุด และวาด confusion matrix เก็บเป็นรูป

confusion matrix อ่านยังไง
    แถว = คำตอบจริง   คอลัมน์ = คำตอบที่โมเดลทาย
    ตัวเลขบนเส้นทแยงมุม = ทายถูก  นอกเส้นทแยงมุม = ทายผิด และผิดเป็นคลาสอะไร
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")          # โหมดไม่เปิดหน้าต่าง เซฟเป็นไฟล์อย่างเดียว
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from config import OUTPUT_DIR


def run():
    X_test = np.load(OUTPUT_DIR / "X_test.npy")
    y_test = np.load(OUTPUT_DIR / "y_test.npy")
    model = joblib.load(OUTPUT_DIR / "svm_model.pkl")

    with open(OUTPUT_DIR / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    y_pred = model.predict(X_test)

    print(f"\nAccuracy = {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=classes))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=classes, cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix - SVM ({model.kernel} kernel)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"บันทึกรูปลง {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    run()
