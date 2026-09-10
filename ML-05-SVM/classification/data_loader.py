"""
data_loader.py
ขั้นที่ 1 : อ่านรูปจากโฟลเดอร์ แปลงเป็นตัวเลข แล้วบันทึกเป็น features.npy / labels.npy

หลักคิด : SVM รับได้เฉพาะตัวเลข รูป 1 ใบจึงต้องถูกแปลงเป็น "แถวตัวเลขยาว 1 แถว"
          รูป 64x64 ขาวดำ -> ตัวเลข 4096 ตัว เรียงต่อกันเป็นแถวเดียว
"""

import json
import numpy as np
from PIL import Image

from config import DATA_DIR, OUTPUT_DIR, IMG_SIZE, GRAYSCALE, MAX_PER_CLASS

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_classes(data_dir):
    """หาชื่อคลาสจากชื่อโฟลเดอร์ย่อย เช่น ['Cat', 'Dog']"""
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์คลาสใน {data_dir}")
    return classes


def image_to_vector(path):
    """อ่านรูป 1 ใบ -> ย่อขนาด -> แปลงเป็นแถวตัวเลข 1 แถว"""
    img = Image.open(path)
    img = img.convert("L" if GRAYSCALE else "RGB")   # L = ขาวดำ
    img = img.resize(IMG_SIZE)
    # หาร 255 เพื่อบีบค่าพิกเซลจาก 0-255 ให้อยู่ในช่วง 0-1
    return np.asarray(img, dtype=np.float32).flatten() / 255.0


def load_dataset():
    classes = find_classes(DATA_DIR)
    print(f"พบคลาสทั้งหมด {len(classes)} คลาส : {classes}")

    features, labels = [], []
    skipped = 0

    for label_index, class_name in enumerate(classes):
        files = [p for p in sorted((DATA_DIR / class_name).iterdir())
                 if p.suffix.lower() in VALID_EXT]
        if MAX_PER_CLASS:
            files = files[:MAX_PER_CLASS]

        loaded = 0
        for path in files:
            try:
                features.append(image_to_vector(path))
                labels.append(label_index)
                loaded += 1
            except Exception:
                # ชุด PetImages มีไฟล์เสียปนอยู่จริง ต้องข้ามไม่ให้โปรแกรมตาย
                skipped += 1

        print(f"  {class_name:<15} โหลดสำเร็จ {loaded} ใบ")

    if skipped:
        print(f"  ข้ามไฟล์เสีย {skipped} ใบ")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, classes


def run():
    X, y, classes = load_dataset()

    np.save(OUTPUT_DIR / "features.npy", X)
    np.save(OUTPUT_DIR / "labels.npy", y)
    with open(OUTPUT_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print(f"\nรูปทั้งหมด {X.shape[0]} ใบ  แต่ละใบมี {X.shape[1]} feature")
    print(f"บันทึกลง {OUTPUT_DIR}")
    return X, y, classes


if __name__ == "__main__":
    run()
