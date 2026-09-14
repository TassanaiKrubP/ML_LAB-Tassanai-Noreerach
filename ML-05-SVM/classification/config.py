"""
config.py
ศูนย์รวมค่าตั้งต้นทั้งหมดของโปรเจกต์ แก้ที่นี่ที่เดียว ไฟล์อื่นดึงไปใช้ต่อ
"""

from pathlib import Path

# ---------- ตำแหน่งโฟลเดอร์ ----------
# BASE_DIR ชี้ไปที่ ML-05-SVM/  (ถอยขึ้นจาก classification/ หนึ่งชั้น)
BASE_DIR = Path(__file__).resolve().parent.parent

# โฟลเดอร์ข้อมูลรูปภาพ  ข้างในต้องมีโฟลเดอร์ย่อย 1 โฟลเดอร์ = 1 คลาส
# ถ้าใช้ชุด Bird/Drone/Airplane ให้เปลี่ยนเป็น BASE_DIR / "BirdVsDroneVsAirplane"
DATA_DIR = BASE_DIR / "BirdVsDroneVsAirplane"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- การเตรียมรูป ----------
IMG_SIZE = (64, 64)      # ย่อรูปทุกใบให้เท่ากัน  64x64 = 4096 พิกเซล
GRAYSCALE = True         # True = แปลงเป็นขาวดำ ทำให้ feature ลดเหลือ 1 ใน 3
MAX_PER_CLASS = 300     # จำกัดจำนวนรูปต่อคลาส กันไม่ให้ SVM เทรนนานเกินไป
                         # ใส่ None ถ้าต้องการใช้รูปทั้งหมด

# ---------- การแบ่งข้อมูลและโมเดล ----------
TEST_SIZE = 0.2          # แบ่ง 20% ไว้ทดสอบ
RANDOM_STATE = 42        # ตัวเลขล็อกการสุ่ม ให้ผลลัพธ์ซ้ำเดิมได้ทุกครั้งที่รัน
PCA_COMPONENTS = 100     # บีบมิติเหลือ 100 ใส่ None ถ้าไม่ต้องการใช้ PCA
KERNELS = ["linear", "poly", "rbf"]   # 3 kernel ตามที่ใบงานกำหนด
MAX_ITER = 200_000       # ขีดจำกัดจำนวนรอบวน กันไม่ให้ linear kernel วนไม่จบ
CACHE_SIZE = 1000        # หน่วยความจำที่ให้ SVM ใช้พักข้อมูล (MB) ช่วยให้เร็วขึ้น