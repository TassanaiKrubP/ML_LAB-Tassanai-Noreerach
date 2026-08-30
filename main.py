"""
ตัวรันรวม - เรียก main.py ของทุกโฟลเดอร์ตามลำดับเลขหน้าชื่อโฟลเดอร์

ใช้โครงสร้างเดียวกับ repo ของอาจารย์ (ML-02-Data Preprocessing/main.py)
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOPICS = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and d[0].isdigit()
)


def main():
    # สร้างข้อมูลดิบก่อน ถ้ายังไม่มีไฟล์
    if not os.path.exists(os.path.join(BASE_DIR, "f1_drivers_raw.csv")):
        print("ยังไม่มี f1_drivers_raw.csv กำลังสร้างให้...")
        subprocess.run([sys.executable, "make_raw_data.py"], cwd=BASE_DIR)

    for topic in TOPICS:
        print("\n" + "--" * 30)
        print(f"-- {topic}")
        print("--" * 30 + "\n")

        subprocess.run([sys.executable, "main.py"],
                       cwd=os.path.join(BASE_DIR, topic))


if __name__ == "__main__":
    main()
