"""
common.py
ค่าตั้งต้นที่ทุกขั้น (01 -> 05) ใช้ร่วมกัน แก้ที่นี่ที่เดียว
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- ข้อมูล ----------
# ต้องการเปลี่ยน dataset : แก้ RAW_DATA_DIR และ CLASSES ให้ตรงกับไฟล์ XML ชุดใหม่
RAW_DATA_DIR = os.path.join(ROOT_DIR, "face-mask-data")          # images/ + annotations/ (PASCAL VOC)
YOLO_DATA_DIR = os.path.join(ROOT_DIR, "yolo-dataset")           # ผลแปลงเป็น YOLO format
DATA_YAML = os.path.join(YOLO_DATA_DIR, "data.yaml")
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

# ---------- การแบ่งข้อมูล ----------
SPLIT_RATIO = {"train": 0.7, "val": 0.2, "test": 0.1}
RANDOM_STATE = 42

# ---------- โมเดล ----------
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
PRETRAINED = os.path.join(WEIGHTS_DIR, "yolo11n.pt")   # YOLO11 nano ฝึกบน COCO มาแล้ว (80 คลาส)
TRAIN_DIR = os.path.join(ROOT_DIR, "03-train-yolo", "outputs", "experiments")
EVAL_DIR = os.path.join(ROOT_DIR, "04-evaluate", "outputs")
BEST_RUN_FILE = os.path.join(EVAL_DIR, "best_run.json")

# ---------- การทดลอง (ตามใบงาน : epoch, ขนาดรูป, batch) ----------
RUNS = {
    "e10":      {"group": "epochs", "epochs": 10, "imgsz": 640, "batch": 16},
    "e30":      {"group": "epochs", "epochs": 30, "imgsz": 640, "batch": 16},
    "e60":      {"group": "epochs", "epochs": 60, "imgsz": 640, "batch": 16},
    "s320_b8":  {"group": "config", "epochs": 30, "imgsz": 320, "batch": 8},
    "s320_b16": {"group": "config", "epochs": 30, "imgsz": 320, "batch": 16},
    "s640_b8":  {"group": "config", "epochs": 30, "imgsz": 640, "batch": 8},
}
# s640_b16 = e30 (ค่าเดียวกันทุกอย่าง) ใช้ผลร่วมกัน ไม่เทรนซ้ำ
ALIASES = {"s640_b16": "e30"}

CONF_THRESHOLD = 0.25   # แสดงกรอบเมื่อความมั่นใจ >= 25%


def best_weights(run_name):
    return os.path.join(TRAIN_DIR, run_name, "weights", "best.pt")


# ---------- อ่าน / เขียนรูป ----------
# cv2.imread / cv2.imwrite บน Windows เปิด path ที่มีตัวอักษรไทย (เช่น D:\วา\...) ไม่ได้
# จึงอ่านไฟล์เป็น bytes ด้วย numpy ก่อน แล้วให้ OpenCV ถอดรหัสจาก bytes แทน
def imread(path):
    import cv2
    import numpy as np
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    import cv2
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)
    return ok
