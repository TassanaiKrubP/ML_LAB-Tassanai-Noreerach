"""
data_loader.py
ขั้นที่ 1 : สำรวจโฟลเดอร์ หาไฟล์รูปที่เปิดได้จริง และข้ามไฟล์เสีย

ไฟล์นี้ยังไม่แปลงรูปเป็นตัวเลข ทำแค่ "คัดไฟล์ที่ใช้ได้" กับ "จดว่ามีคลาสอะไรบ้าง"
งานแปลงเป็นตัวเลขอยู่ใน preprocessing.py
"""

import json

from PIL import Image

from config import DATA_DIR, OUTPUT_DIR, MAX_PER_CLASS

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_classes(data_dir):
    """ชื่อโฟลเดอร์ย่อย = ชื่อคลาส"""
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์คลาสใน {data_dir}")
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
    classes = find_classes(DATA_DIR)
    print(f"พบคลาสทั้งหมด {len(classes)} คลาส : {classes}")

    paths, labels, skipped = [], [], 0

    for label_index, class_name in enumerate(classes):
        files = [p for p in sorted((DATA_DIR / class_name).iterdir())
                 if p.suffix.lower() in VALID_EXT]

        good = []
        for path in files:
            if is_readable(path):
                good.append(path)
            else:
                skipped += 1
            if MAX_PER_CLASS and len(good) >= MAX_PER_CLASS:
                break

        paths.extend(good)
        labels.extend([label_index] * len(good))
        print(f"  {class_name:<15} ใช้ได้ {len(good)} ใบ")

    if skipped:
        print(f"  ข้ามไฟล์เสีย {skipped} ใบ")

    with open(OUTPUT_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print(f"รวมรูปที่ใช้ได้ {len(paths)} ใบ")
    return paths, labels, classes


if __name__ == "__main__":
    run()
