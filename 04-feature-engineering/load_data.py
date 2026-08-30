import os
import sys

import pandas as pd

# Part 4 ใช้ข้อมูลที่ Part 3 ทำความสะอาดไว้แล้ว ไม่ใช่ข้อมูลดิบ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "f1_drivers_clean.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    """อ่านข้อมูลที่ผ่านการทำความสะอาดจาก Part 3"""

    if not os.path.exists(CSV_PATH):
        print("ไม่พบ f1_drivers_clean.csv กรุณารัน 03-data-cleaning/main.py ก่อน")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    return df
