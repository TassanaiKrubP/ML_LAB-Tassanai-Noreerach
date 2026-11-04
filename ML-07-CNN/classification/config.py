"""
config.py
ศูนย์รวมค่าตั้งต้นทั้งหมดของ Lab 7 แก้ที่นี่ที่เดียว
"""

import os

# ---------- ตำแหน่งโฟลเดอร์ ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ใช้รูปชุดเดียวกับ Lab 6 โดยชี้ไปที่โฟลเดอร์เดิม ไม่คัดลอกซ้ำ
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "ML-06-NN", "BirdVsDroneVsAirplane")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "experiments")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DATASET_FILE = os.path.join(OUTPUT_DIR, "dataset.npz")   # รูปทั้งหมด (uint8)
SPLIT_FILE = os.path.join(OUTPUT_DIR, "splits.npz")      # train / val / test (uint8)
for folder in (OUTPUT_DIR, EXPERIMENT_DIR, LOG_DIR):
    os.makedirs(folder, exist_ok=True)

# ---------- การเตรียมรูป ----------
IMG_SIZE = 96            # ย่อรูปเป็น 96x96 สี RGB (พอดีกับ CPU + RAM 8 GB)
# Bird มีแค่ 582 ใบ จำกัดทุกคลาสให้เท่ากันแบบ Lab 6 เพื่อไม่ให้โมเดลเอียงไปทางคลาสที่รูปเยอะ
MAX_PER_CLASS = 582

# ---------- การแบ่งข้อมูล ----------
TEST_SIZE = 0.20         # 20% ของทั้งหมด เก็บไว้สอบครั้งสุดท้าย
VAL_SIZE = 0.10          # 10% ของทั้งหมด ใช้ตรวจทุกจบ epoch
RANDOM_STATE = 42

# ---------- การเทรน ----------
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 15                          # รอบเทรนของการทดลองเทียบโครงสร้าง
EPOCH_RUN_EPOCHS = 30                # การทดลองเทียบจำนวน epoch เทรนยาวรอบเดียว
EPOCH_CHECKPOINTS = [5, 10, 15, 20, 30]
EARLY_STOP_PATIENCE = 5
