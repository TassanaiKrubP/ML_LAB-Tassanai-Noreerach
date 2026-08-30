"""
Part 3 - Data Cleaning

ทำความสะอาดข้อมูลตามลำดับนี้
    1. Data Type Conversion   - แปลงชนิดข้อมูลให้ถูกต้องก่อน ไม่งั้นคำนวณอะไรไม่ได้เลย
    2. Duplicate Removal      - ลบแถวซ้ำก่อนเติมค่า ไม่งั้นค่าเฉลี่ยจะเพี้ยนตามแถวซ้ำ
    3. Incorrect Data         - เปลี่ยนค่าที่เป็นไปไม่ได้ให้เป็น NaN
    4. Missing Value Handling - แล้วค่อยเติมค่าที่หายไปทั้งหมดในรอบเดียว

ลำดับสำคัญมาก เหมือนล้างผักก่อนหั่น ถ้าสลับกันงานจะเพิ่มโดยไม่จำเป็น
"""

import os

import numpy as np
import pandas as pd

from load_data import load_data, OUTPUT_DIR, BASE_DIR
from plot_result import plot_before_after, plot_mean_vs_median

CLEAN_PATH = os.path.join(BASE_DIR, "..", "f1_drivers_clean.csv")


def main():
    print("=" * 55)
    print("03 - Data Cleaning")
    print("=" * 55)

    df = load_data()
    original = df.copy()

    # ------------------------------------------------------------------
    print("\n[1] Data Type Conversion")
    # career_points เก็บมาเป็นข้อความเพราะมีลูกน้ำคั่นหลักพัน เช่น "4,862"
    print(f"ก่อนแปลง career_points มีชนิด: {df['career_points'].dtype}")
    df["career_points"] = pd.to_numeric(
        df["career_points"].astype(str).str.replace(",", ""), errors="coerce"
    )
    print(f"หลังแปลง career_points มีชนิด: {df['career_points'].dtype}")

    # ------------------------------------------------------------------
    print("\n[2] Duplicate Removal")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"ลบแถวซ้ำไป {before - len(df)} แถว เหลือ {len(df)} แถว")

    # ------------------------------------------------------------------
    print("\n[3] Incorrect Data Correction")

    # 3.1 ชื่อทีมพิมพ์ไม่เหมือนกัน -> ตัดช่องว่างหัวท้ายแล้วจัดรูปแบบตัวพิมพ์ใหม่
    print(f"ก่อนแก้ มีชื่อทีมทั้งหมด {df['team'].nunique()} แบบ")
    df["team"] = df["team"].str.strip().str.title()

    # str.title() ทำให้ McLaren กลายเป็น Mclaren จึงต้องแก้เป็นกรณีพิเศษ
    df["team"] = df["team"].replace({"Mclaren": "McLaren"})
    print(f"หลังแก้ เหลือ {df['team'].nunique()} แบบ -> {sorted(df['team'].dropna().unique())}")

    # 3.2 ค่าที่เป็นไปไม่ได้ -> เปลี่ยนเป็น NaN เพื่อให้ขั้นตอนถัดไปเติมค่าให้
    rules = {
        # ชุดข้อมูลนี้มีทั้งนักแข่งปัจจุบันและตำนานที่เลิกแข่งไปแล้ว (อายุ 80 กว่า)
        # จึงตั้งช่วงให้กว้าง ดักเฉพาะค่าที่เป็นไปไม่ได้จริง ๆ
        "age": (16, 95),
        "height_cm": (150, 210),
        "career_wins": (0, 200),
    }

    for column, (low, high) in rules.items():
        bad = ~df[column].between(low, high) & df[column].notna()
        if bad.any():
            print(f"  {column}: พบค่าผิดปกติ {bad.sum()} ค่า -> {list(df.loc[bad, column])}")
        df.loc[bad, column] = np.nan

    # ------------------------------------------------------------------
    print("\n[4] Missing Value Handling")
    missing = df.isna().sum()
    print(missing[missing > 0])

    # ตัวเลข -> เติมด้วย median, ข้อความ -> เติมด้วย "Unknown"
    for column in ["age", "height_cm", "career_wins"]:
        df[column] = df[column].fillna(df[column].median())

    df["team"] = df["team"].fillna("Unknown")

    print(f"\nหลังเติมค่า เหลือค่าที่หายไป {int(df.isna().sum().sum())} ช่อง")

    # ------------------------------------------------------------------
    print("\n[5] Compare Mean vs Median")

    # เอาคอลัมน์ age ที่ยังมีรูโหว่กลับมาเพื่อเปรียบเทียบสองวิธีอย่างเป็นธรรม
    raw_age = original["age"].copy()
    raw_age = raw_age.where(raw_age.between(16, 95))

    filled_mean = raw_age.fillna(raw_age.mean())
    filled_median = raw_age.fillna(raw_age.median())

    print(f"{'วิธี':<24}{'Mean':>10}{'Median':>10}{'Std':>10}")
    print(f"{'ก่อนเติม (มี NaN)':<24}{raw_age.mean():>10.2f}{raw_age.median():>10.2f}{raw_age.std():>10.2f}")
    print(f"{'เติมด้วย mean':<24}{filled_mean.mean():>10.2f}{filled_mean.median():>10.2f}{filled_mean.std():>10.2f}")
    print(f"{'เติมด้วย median':<24}{filled_median.mean():>10.2f}{filled_median.median():>10.2f}{filled_median.std():>10.2f}")
    print("\nสรุป: ทั้งสองวิธีทำให้ค่า std ลดลง เพราะเราไปยัดค่าเดิมซ้ำ ๆ ลงตรงกลาง")
    print("      แต่ median ทนต่อค่าสุดโต่งมากกว่า จึงเลือกใช้ median ในใบงานนี้")

    # ------------------------------------------------------------------
    print("\n[6] วาดกราฟ")
    plot_before_after(original, df, OUTPUT_DIR)
    plot_mean_vs_median(raw_age, filled_mean, filled_median, OUTPUT_DIR)

    # ------------------------------------------------------------------
    print("\n[7] บันทึกข้อมูลที่สะอาดแล้ว")
    df.to_csv(CLEAN_PATH, index=False)
    print(f"บันทึกแล้ว: f1_drivers_clean.csv ({len(df)} แถว)")


if __name__ == "__main__":
    main()
