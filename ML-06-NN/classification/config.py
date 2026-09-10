"""
config.py
ศูนย์รวมค่าตั้งต้นทั้งหมดของ Lab 6 แก้ที่นี่ที่เดียว
"""

from pathlib import Path

# ---------- ตำแหน่งโฟลเดอร์ ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "BirdVsDroneVsAirplane"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- การเตรียมรูป ----------
IMG_SIZE = (64, 64)      # ย่อรูปทุกใบให้เท่ากัน
CHANNELS = 3             # 3 = RGB (Lab 5 ใช้ขาวดำ Lab 6 ใช้สีตามใบงาน)
MAX_PER_CLASS = 2000     # จำกัดจำนวนรูปต่อคลาส  ใส่ None ถ้าใช้ทั้งหมด

# ---------- การแบ่งข้อมูล 3 ส่วน ----------
# Lab 5 แบ่ง 2 ส่วน  Lab 6 เพิ่ม validation เข้ามาเพื่อเฝ้าดูอาการ overfitting
TEST_SIZE = 0.15         # 15% เก็บไว้สอบปลายทาง ไม่แตะจนกว่าจะจบ
VAL_SIZE = 0.15          # 15% ใช้ตรวจสอบระหว่างเทรนทุก epoch
RANDOM_STATE = 42

# ---------- การเทรน ----------
EPOCHS = 40              # เทรนยาวครั้งเดียว แล้วอ่านผลย้อนหลังที่ epoch ต่าง ๆ
BATCH_SIZE = 32          # จำนวนรูปที่ป้อนเข้าโมเดลต่อการปรับน้ำหนัก 1 ครั้ง
LEARNING_RATE = 0.001

# จุดที่จะเอามาเปรียบเทียบว่า "เทรนกี่ epoch ถึงพอ" ตามที่ใบงานกำหนด
EPOCH_CHECKPOINTS = [5, 10, 20, 30, 40]

# ---------- โครงสร้างเครือข่ายที่จะเปรียบเทียบ ----------
# hidden = จำนวนนิวรอนในแต่ละชั้นซ่อน  เช่น [256, 128] คือ 2 ชั้น ชั้นแรก 256 ตัว
CONFIGURATIONS = [
    {"name": "1-hidden-128",       "hidden": [128],            "dropout": 0.0},
    {"name": "2-hidden-256-128",   "hidden": [256, 128],       "dropout": 0.3},
    {"name": "3-hidden-512-256-128", "hidden": [512, 256, 128], "dropout": 0.3},
]
