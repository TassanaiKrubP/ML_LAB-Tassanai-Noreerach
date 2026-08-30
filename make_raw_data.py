"""
สร้างไฟล์ f1_drivers_raw.csv จากไฟล์ต้นฉบับ f1_drivers.csv

ข้อมูลต้นฉบับสะอาดเกินไป จึงจงใจใส่ปัญหาลงไปเพื่อให้มีอะไรให้ทำความสะอาด
ในใบงานนี้ (ใช้ seed คงที่ ผลลัพธ์จึงเหมือนเดิมทุกครั้งที่รัน)

ปัญหาที่ใส่ลงไป
  1. Missing values   -> age, height_cm, team
  2. Duplicate rows   -> คัดลอกบางแถวมาต่อท้าย
  3. Incorrect data   -> อายุติดลบ, อายุ 999, ส่วนสูงผิดหน่วย
  4. Inconsistent     -> ชื่อทีมพิมพ์ไม่เหมือนกัน (ตัวพิมพ์เล็ก/มีช่องว่าง)
  5. Wrong data type  -> career_points เก็บเป็นข้อความมีลูกน้ำ เช่น "4,862"
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(BASE_DIR, "f1_drivers.csv")
RAW_PATH = os.path.join(BASE_DIR, "f1_drivers_raw.csv")


def main():
    rng = np.random.default_rng(42)
    df = pd.read_csv(SOURCE_PATH)
    print(f"ต้นฉบับ: {len(df)} แถว {len(df.columns)} คอลัมน์")

    # ---- 1. เจาะรูให้เป็น missing values ----
    for column, ratio in [("age", 0.10), ("height_cm", 0.06), ("team", 0.04)]:
        holes = rng.choice(len(df), size=int(len(df) * ratio), replace=False)
        df.loc[holes, column] = np.nan

    # ---- 2. ทำให้ชื่อทีมพิมพ์ไม่สม่ำเสมอ ----
    messy_team = {0: " Ferrari ", 1: "red bull", 5: "MERCEDES", 9: "Ferrari  "}
    for row, value in messy_team.items():
        df.loc[row, "team"] = value

    # ---- 3. ใส่ค่าที่ผิดปกติ (outlier / ค่าที่เป็นไปไม่ได้) ----
    df.loc[3, "age"] = -38          # อายุติดลบ
    df.loc[7, "age"] = 999          # อายุเกินจริง
    df.loc[11, "height_cm"] = 17.8  # ส่วนสูงผิดหน่วย (น่าจะ 178)
    df.loc[15, "career_wins"] = -2  # จำนวนชนะติดลบ

    # ---- 4. เปลี่ยน career_points ให้เป็นข้อความมีลูกน้ำ ----
    df["career_points"] = df["career_points"].apply(lambda x: f"{x:,}")

    # ---- 5. คัดลอกแถวซ้ำมาต่อท้าย ----
    duplicated = df.iloc[[2, 4, 6, 20, 33]].copy()
    df = pd.concat([df, duplicated], ignore_index=True)

    df.to_csv(RAW_PATH, index=False)
    print(f"บันทึกแล้ว: f1_drivers_raw.csv ({len(df)} แถว)")


if __name__ == "__main__":
    main()
