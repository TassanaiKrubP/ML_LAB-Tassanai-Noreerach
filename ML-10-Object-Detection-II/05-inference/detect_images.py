"""
05-inference/detect_images.py
ใช้โมเดลที่ดีที่สุดตรวจจับหน้ากากในรูป test  แต่ละกรอบมี ชื่อคลาส + ความมั่นใจ (confidence)

    python detect_images.py                       รูป test ทั้งหมด
    python detect_images.py --source รูป.jpg      รูปของเราเอง (ไฟล์หรือโฟลเดอร์)

ผลลัพธ์
    outputs/images/*.jpg      รูปพร้อมกรอบ (12 ใบแรก)
    outputs/detections.csv    ทุกกรอบที่เจอ : รูป, คลาส, confidence, พิกัด
    outputs/detections_grid.png
"""

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from common import YOLO_DATA_DIR, BEST_RUN_FILE, CONF_THRESHOLD, best_weights, imread, imwrite

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
N_SAVE = 12


def load_best_model():
    with open(BEST_RUN_FILE, encoding="utf-8") as f:
        best = json.load(f)
    print(f"  โมเดล : {best['run']} (imgsz {best['imgsz']})")
    return YOLO(best_weights(best["run"])), int(best["imgsz"])


def list_images(source):
    if os.path.isdir(source):
        return [os.path.join(source, f) for f in sorted(os.listdir(source))
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
    return [source]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.path.join(YOLO_DATA_DIR, "images", "test"))
    args = parser.parse_args()
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

    print("[Step 5a] Detecting masks in test images...")
    model, imgsz = load_best_model()
    files = list_images(args.source)

    rows, grid = [], []
    for i, path in enumerate(files):
        img = imread(path)
        r = model.predict(img, imgsz=imgsz, conf=CONF_THRESHOLD, device=0, verbose=False)[0]
        name = os.path.basename(path)
        for box, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            rows.append([name, model.names[int(c)], round(s, 3)] + [round(v, 1) for v in box])
        if i < N_SAVE:
            annotated = r.plot()     # วาดกรอบ + ชื่อคลาส + confidence
            imwrite(os.path.join(OUT_DIR, "images", os.path.splitext(name)[0] + ".jpg"), annotated)
            if len(grid) < 8:
                grid.append((name, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), len(r.boxes)))

    with open(os.path.join(OUT_DIR, "detections.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)
    print(f"  ตรวจ {len(files)} รูป เจอทั้งหมด {len(rows)} กรอบ  บันทึก {min(N_SAVE, len(files))} รูปใน outputs/images/")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, img, n) in zip(axes.ravel(), grid):
        ax.imshow(img)
        ax.set_title(f"{name}  ({n} boxes)", fontsize=9)
    fig.suptitle(f"Detections on test images (conf >= {CONF_THRESHOLD})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "detections_grid.png"), dpi=110)
    plt.close(fig)
    print("  บันทึก detections_grid.png และ detections.csv")


if __name__ == "__main__":
    main()
