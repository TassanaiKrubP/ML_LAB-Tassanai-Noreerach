"""
config.py
ศูนย์รวมค่าตั้งต้นทั้งหมดของ Lab 8 แก้ที่นี่ที่เดียว
ใช้รูป วิธีลบรูปซ้ำ และ seed เดียวกับ Lab 7 ชุด test จึงเป็นชุดเดียวกัน เทียบคะแนนกันได้ตรง ๆ
"""

import os

# ---------- ตำแหน่งโฟลเดอร์ ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "ML-06-NN", "BirdVsDroneVsAirplane")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "experiments")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DATASET_FILE = os.path.join(OUTPUT_DIR, "dataset.npz")
SPLIT_FILE = os.path.join(OUTPUT_DIR, "splits.npz")
# ผลของ Lab 7 ใช้วาดกราฟเทียบทุกแล็ป
LAB7_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "ML-07-CNN", "classification", "outputs")
for folder in (OUTPUT_DIR, EXPERIMENT_DIR, LOG_DIR):
    os.makedirs(folder, exist_ok=True)

# ---------- การเตรียมรูป ----------
IMG_SIZE = 96            # เก็บรูปขนาด 96x96 (ขนาดที่ใช้กับ transfer learning)
SCRATCH_IMG_SIZE = 64    # VGG เทรนเองตั้งแต่ศูนย์ย่อเหลือ 64x64 ในโมเดล (CPU เทรนไหว)
MAX_PER_CLASS = 582      # เหมือน Lab 7

# ---------- การแบ่งข้อมูล ----------
TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_STATE = 42

# ---------- การเทรน (ตามแบบอาจารย์ แต่ลด batch ให้พอกับ RAM 8 GB) ----------
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
FINETUNE_LEARNING_RATE = 1e-5
EPOCHS = 20
FINETUNE_EPOCHS = 5
EPOCH_RUN_EPOCHS = 30
EPOCH_CHECKPOINTS = [5, 10, 15, 20, 30]
EARLY_STOP_PATIENCE = 8
REDUCE_LR_PATIENCE = 4

# ผล test accuracy ของแล็ปก่อน ๆ (Lab 4, 5 จาก README / Lab 6 คำนวณจากโมเดลที่บันทึกไว้)
PREVIOUS_LABS = {"KNN (Lab 4)": 0.7167, "SVM (Lab 5)": 0.7648, "NN (Lab 6)": 0.7672}
