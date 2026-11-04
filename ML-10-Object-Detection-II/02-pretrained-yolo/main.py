"""
02-pretrained-yolo/main.py
ขั้นที่ 2 : โหลด YOLO11n ที่ฝึกบน COCO มาแล้ว (80 คลาส เช่น person, car, dog) แล้วลองกับรูปหน้ากาก

ผลที่คาด : โมเดลเจอ "person" ได้ แต่ไม่รู้จัก with_mask / without_mask เพราะ COCO ไม่มีคลาสเหล่านี้
ต้องนำไปเทรนต่อ (fine-tune) กับข้อมูลของเราในขั้นที่ 3 = transfer learning แบบเดียวกับ Lab 8
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from common import PRETRAINED, WEIGHTS_DIR, YOLO_DATA_DIR, imread

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("[Step 1] Loading pretrained YOLO11n (COCO)...")
    model = YOLO(PRETRAINED)          # ไม่มีไฟล์จะดาวน์โหลดให้อัตโนมัติ
    names = model.names
    print(f"  โมเดลรู้จัก {len(names)} คลาส เช่น {list(names.values())[:8]} ...")
    has_mask = [n for n in names.values() if "mask" in n]
    print(f"  มีคลาสเกี่ยวกับ mask หรือไม่ : {has_mask if has_mask else 'ไม่มี'}")

    print("[Step 2] Running it on 4 test images...")
    img_dir = os.path.join(YOLO_DATA_DIR, "images", "test")
    files = sorted(os.listdir(img_dir))[:4]
    lines = []
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    for ax, name in zip(axes, files):
        img = imread(os.path.join(img_dir, name))
        result = model.predict(img, conf=0.25, verbose=False)[0]
        found = [f"{names[int(c)]} {s:.2f}" for c, s in zip(result.boxes.cls, result.boxes.conf)]
        lines.append(f"{name}: {', '.join(found) if found else '(nothing)'}")
        print("  " + lines[-1])
        ax.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Pretrained COCO model before fine-tuning: finds 'person', but has no mask classes",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pretrained_predictions.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "pretrained_predictions.txt"), "w", encoding="utf-8") as f:
        f.write(f"COCO classes: {len(names)}\nmask classes in COCO: {has_mask or 'none'}\n\n")
        f.write("\n".join(lines) + "\n")
    print("  บันทึก outputs/pretrained_predictions.png และ .txt")


if __name__ == "__main__":
    main()
