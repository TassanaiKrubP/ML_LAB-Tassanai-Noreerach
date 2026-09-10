r"""
test_svm.py
นำโมเดลที่เทรนเสร็จแล้วมาทายรูปใบใหม่ที่โมเดลไม่เคยเห็น

วิธีใช้ (PowerShell)
    python test_svm.py ..\PetImages\Cat\5.jpg

สำคัญ : รูปใบใหม่ต้องผ่านขั้นตอนเตรียมข้อมูล "ชุดเดียวกัน" กับตอนเทรน
        คือย่อขนาดเท่ากัน แล้ว transform ด้วย scaler.pkl ตัวเดิม
        ถ้าข้ามขั้นนี้ โมเดลจะทายมั่วทันที
"""

import json
import sys

import joblib

from config import OUTPUT_DIR
from data_loader import image_to_vector


def predict(image_path):
    pipeline = joblib.load(OUTPUT_DIR / "scaler.pkl")
    model = joblib.load(OUTPUT_DIR / "svm_model.pkl")
    with open(OUTPUT_DIR / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    vector = image_to_vector(image_path).reshape(1, -1)   # ทำให้เป็นตาราง 1 แถว
    vector = pipeline.transform(vector)
    label_index = model.predict(vector)[0]

    return classes[label_index]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ใส่ path ของรูปด้วย เช่น  python test_svm.py ..\\PetImages\\Cat\\5.jpg")
        sys.exit(1)

    path = sys.argv[1]
    print(f"รูป : {path}")
    print(f"โมเดลทายว่า : {predict(path)}")
