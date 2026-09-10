"""
preprocessing.py
ขั้นที่ 2 : ย่อรูปให้เท่ากัน แปลงเป็น RGB แล้วปรับสเกลค่าพิกเซล

เรื่องลำดับสี BGR กับ RGB
    ไลบรารี OpenCV อ่านรูปออกมาเป็นลำดับ BGR (น้ำเงิน-เขียว-แดง) จึงต้องแปลงเป็น RGB
    แต่โค้ดนี้ใช้ Pillow ซึ่งอ่านออกมาเป็น RGB อยู่แล้ว จึงไม่ต้องสลับช่องสี
    ใช้ .convert("RGB") เพื่อบังคับให้ทุกรูปมี 3 ช่องสีเท่ากัน
    (รูปขาวดำหรือรูป PNG แบบโปร่งใสที่ปนมาจะถูกแปลงให้เป็น RGB ด้วย)

การ Standardize ของรูปภาพ
    ค่าพิกเซลเดิมอยู่ในช่วง 0-255 หารด้วย 255 ให้เหลือ 0-1
    Neural Network ปรับน้ำหนักด้วยการคูณสะสมหลายชั้น ถ้าตัวเลขนำเข้าใหญ่เกินไป
    ค่าที่คำนวณได้จะบานปลายจนเทรนไม่ลง การบีบให้อยู่ในช่วงแคบจึงจำเป็น
"""

import numpy as np
from PIL import Image

from config import OUTPUT_DIR, IMG_SIZE

import data_loader


def image_to_array(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.asarray(img, dtype=np.float32) / 255.0


def run():
    paths, labels, classes = data_loader.run()

    X = np.stack([image_to_array(p) for p in paths]).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    np.save(OUTPUT_DIR / "features.npy", X)
    np.save(OUTPUT_DIR / "labels.npy", y)

    print(f"\nรูปทั้งหมด {X.shape[0]} ใบ  ขนาดต่อใบ {X.shape[1]}x{X.shape[2]}x{X.shape[3]}")
    print(f"ค่าพิกเซลอยู่ในช่วง {X.min():.2f} ถึง {X.max():.2f}")
    return X, y, classes


if __name__ == "__main__":
    run()
