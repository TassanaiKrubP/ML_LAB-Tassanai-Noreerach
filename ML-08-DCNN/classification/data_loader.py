"""
data_loader.py
ขั้นที่ 1 : สำรวจโฟลเดอร์ หาไฟล์รูปที่เปิดได้จริง และข้ามไฟล์เสีย

ไฟล์นี้ยังไม่แปลงรูปเป็นตัวเลข ทำแค่ "คัดไฟล์ที่ใช้ได้" กับ "จดว่ามีคลาสอะไรบ้าง"
งานแปลงเป็นตัวเลขอยู่ใน preprocessing.py
"""

import json
import os

from PIL import Image

from config import DATA_PATH, OUTPUT_DIR

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_classes(data_path):
    """ชื่อโฟลเดอร์ย่อย = ชื่อคลาส"""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ข้อมูล {data_path}")
    classes = sorted(d for d in os.listdir(data_path)
                     if os.path.isdir(os.path.join(data_path, d)))
    if not classes:
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์คลาสใน {data_path}")
    return classes


def is_readable(path):
    """เปิดรูปดูจริงเพื่อคัดไฟล์เสียออก"""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def run():
    classes = find_classes(DATA_PATH)
    print(f"พบคลาสทั้งหมด {len(classes)} คลาส : {classes}")

    paths, labels, skipped = [], [], 0
    for label_index, class_name in enumerate(classes):
        folder = os.path.join(DATA_PATH, class_name)
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if os.path.splitext(f)[1].lower() in VALID_EXT]

        good = [p for p in files if is_readable(p)]
        skipped += len(files) - len(good)

        paths.extend(good)
        labels.extend([label_index] * len(good))
        print(f"  {class_name:<15} ใช้ได้ {len(good)} ใบ")

    if skipped:
        print(f"  ข้ามไฟล์เสีย {skipped} ใบ")

    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print(f"รวมรูปที่ใช้ได้ {len(paths)} ใบ")
    return paths, labels, classes


if __name__ == "__main__":
    run()
