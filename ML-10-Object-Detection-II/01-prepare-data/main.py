"""
01-prepare-data/main.py
ขั้นที่ 1 : เตรียม dataset ให้อยู่ในรูปแบบ YOLO

    1  อ่าน XML (PASCAL VOC) ทุกไฟล์ แปลงกรอบเป็น YOLO txt
    2  สุ่มแบ่งรูป train 70% / val 20% / test 10% (seed 42)
    3  วางไฟล์ตามโครงสร้างที่ YOLO ต้องการ แล้วเขียน data.yaml

        yolo-dataset/
        ├── images/train  images/val  images/test
        ├── labels/train  labels/val  labels/test     (ชื่อเดียวกับรูป แต่นามสกุล .txt)
        └── data.yaml

    4  วาดกราฟจำนวนกรอบต่อคลาส และรูปตัวอย่างพร้อมกรอบจริง (ground truth)
"""

import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from common import RAW_DATA_DIR, YOLO_DATA_DIR, DATA_YAML, CLASSES, SPLIT_RATIO, RANDOM_STATE, imread
from voc_to_yolo import parse_voc, to_yolo_lines

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
COLORS = {0: (42, 120, 214), 1: (235, 104, 52), 2: (237, 161, 0)}   # RGB ต่อคลาส


def split_files(names):
    names = sorted(names)
    random.Random(RANDOM_STATE).shuffle(names)
    n_train = int(len(names) * SPLIT_RATIO["train"])
    n_val = int(len(names) * SPLIT_RATIO["val"])
    return {"train": names[:n_train],
            "val": names[n_train:n_train + n_val],
            "test": names[n_train + n_val:]}


def link_or_copy(src, dst):
    """ใช้ hard link (ไม่เปลืองพื้นที่) ถ้าทำไม่ได้ค่อยคัดลอก"""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_all():
    ann_dir = os.path.join(RAW_DATA_DIR, "annotations")
    img_dir = os.path.join(RAW_DATA_DIR, "images")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"ไม่พบ {ann_dir}  ดูวิธีดาวน์โหลดใน README")

    xml_files = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]
    splits = split_files(xml_files)
    counts = {s: [0] * len(CLASSES) for s in splits}
    n_images = {s: len(v) for s, v in splits.items()}

    for split, files in splits.items():
        os.makedirs(os.path.join(YOLO_DATA_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATA_DIR, "labels", split), exist_ok=True)
        for xml_name in files:
            filename, w, h, objects = parse_voc(os.path.join(ann_dir, xml_name))
            lines = to_yolo_lines(objects, w, h, CLASSES)
            for line in lines:
                counts[split][int(line.split()[0])] += 1

            stem = os.path.splitext(filename)[0]
            link_or_copy(os.path.join(img_dir, filename),
                         os.path.join(YOLO_DATA_DIR, "images", split, filename))
            with open(os.path.join(YOLO_DATA_DIR, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    data = {"path": YOLO_DATA_DIR.replace("\\", "/"),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {i: c for i, c in enumerate(CLASSES)}}
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return n_images, counts


def plot_class_distribution(counts):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    splits = list(counts)
    w = 0.26
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for i, split in enumerate(splits):
        xs = [c + (i - 1) * (w + 0.02) for c in range(len(CLASSES))]
        bars = ax.bar(xs, counts[split], w, color=colors[i], label=split)
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("number of boxes")
    ax.set_title("Boxes per class in each split (classes are imbalanced)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=120)
    plt.close(fig)


def draw_gt(image_path, label_path):
    """วาดกรอบจากไฟล์ YOLO txt กลับลงรูป เพื่อพิสูจน์ว่าแปลงพิกัดถูกต้อง"""
    img = cv2.cvtColor(imread(image_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    with open(label_path) as f:
        for line in f:
            c, xc, yc, bw, bh = line.split()
            c, xc, yc, bw, bh = int(c), float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            p1 = (int(xc - bw / 2), int(yc - bh / 2))
            p2 = (int(xc + bw / 2), int(yc + bh / 2))
            cv2.rectangle(img, p1, p2, COLORS[c], 2)
            cv2.putText(img, CLASSES[c], (p1[0], max(10, p1[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[c], 1, cv2.LINE_AA)
    return img


def plot_samples(n=6):
    img_dir = os.path.join(YOLO_DATA_DIR, "images", "train")
    names = sorted(os.listdir(img_dir))[:n]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, name in zip(axes.ravel(), names):
        stem = os.path.splitext(name)[0]
        img = draw_gt(os.path.join(img_dir, name),
                      os.path.join(YOLO_DATA_DIR, "labels", "train", stem + ".txt"))
        ax.imshow(img)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Ground-truth boxes drawn back from YOLO labels  "
                 "(blue = with_mask, orange = without_mask, yellow = mask_weared_incorrect)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sample_gt_boxes.png"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[Step 1] Converting PASCAL VOC XML -> YOLO txt and splitting 70/20/10...")
    n_images, counts = convert_all()
    for split in n_images:
        print(f"  {split:<5} {n_images[split]:>4} รูป  กรอบต่อคลาส {dict(zip(CLASSES, counts[split]))}")
    print(f"  เขียน {DATA_YAML}")

    with open(os.path.join(OUT_DIR, "split_summary.txt"), "w", encoding="utf-8") as f:
        f.write("split,images," + ",".join(CLASSES) + "\n")
        for split in n_images:
            f.write(f"{split},{n_images[split]}," + ",".join(map(str, counts[split])) + "\n")

    print("[Step 2] Plotting class distribution and ground-truth samples...")
    plot_class_distribution(counts)
    plot_samples()
    print("  บันทึก outputs/class_distribution.png, sample_gt_boxes.png, split_summary.txt")


if __name__ == "__main__":
    main()
