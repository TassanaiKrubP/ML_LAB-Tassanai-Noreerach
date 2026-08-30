"""
Part 4 - Feature Engineering

โมเดล ML คำนวณได้แต่ตัวเลข คอลัมน์อย่าง team ที่เป็นข้อความจึงต้องแปลงก่อน
มีสองวิธีหลักที่ให้ผลต่างกันมาก

    Label Encoding    - แทนแต่ละทีมด้วยเลข 0, 1, 2, ...
                        ข้อเสียคือโมเดลอาจเข้าใจผิดว่า 2 > 1 (ทีมนี้ดีกว่าทีมนั้น)
                        ทั้งที่ชื่อทีมไม่มีลำดับอะไรเลย

    One-Hot Encoding  - สร้างคอลัมน์ใหม่ทีมละหนึ่งคอลัมน์ ใส่ 0 หรือ 1
                        ไม่มีปัญหาเรื่องลำดับ แต่คอลัมน์บานปลายถ้ามีหลายทีม
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from load_data import load_data, OUTPUT_DIR
from plot_result import plot_encoding_comparison, plot_scaling_comparison


def main():
    print("=" * 55)
    print("04 - Feature Engineering")
    print("=" * 55)

    df = load_data()

    print(f"\nคอลัมน์ team มีทั้งหมด {df['team'].nunique()} ทีม")
    print(sorted(df["team"].unique()))

    # ------------------------------------------------------------------
    print("\n[1] Label Encoding")
    encoder = LabelEncoder()
    df["team_label"] = encoder.fit_transform(df["team"])

    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    for team, code in mapping.items():
        print(f"  {team:<16} -> {code}")

    print("\nตัวอย่างผลลัพธ์")
    print(df[["name", "team", "team_label"]].head(8))

    # ------------------------------------------------------------------
    print("\n[2] One-Hot Encoding")
    onehot = pd.get_dummies(df["team"], prefix="team").astype(int)
    print(f"สร้างคอลัมน์ใหม่ {onehot.shape[1]} คอลัมน์ จากคอลัมน์เดียว")

    print("\nตัวอย่างผลลัพธ์ (แสดง 5 คอลัมน์แรก)")
    print(pd.concat([df["name"], onehot.iloc[:, :5]], axis=1).head(8))

    df_encoded = pd.concat([df.drop(columns=["team"]), onehot], axis=1)
    print(f"\nขนาดตารางก่อน one-hot: {df.shape}")
    print(f"ขนาดตารางหลัง one-hot: {df_encoded.shape}")

    # ------------------------------------------------------------------
    print("\n[3] Feature Scaling (เพิ่มเติมจากใบงาน)")
    # career_points มีค่าหลักพัน แต่ championships มีค่า 0-7
    # ถ้าไม่ปรับสเกล โมเดลจะให้ความสำคัญกับคอลัมน์ที่ตัวเลขใหญ่มากเกินจริง
    features = ["age", "height_cm", "career_wins", "career_points"]

    standard = pd.DataFrame(
        StandardScaler().fit_transform(df[features]), columns=features
    )
    minmax = pd.DataFrame(
        MinMaxScaler().fit_transform(df[features]), columns=features
    )

    print("\nก่อนปรับสเกล")
    print(df[features].describe().loc[["mean", "std", "min", "max"]].round(2))
    print("\nหลัง StandardScaler (mean = 0, std = 1)")
    print(standard.describe().loc[["mean", "std", "min", "max"]].round(2))
    print("\nหลัง MinMaxScaler (บีบให้อยู่ในช่วง 0 ถึง 1)")
    print(minmax.describe().loc[["mean", "std", "min", "max"]].round(2))

    # ------------------------------------------------------------------
    print("\n[4] วาดกราฟ")
    plot_encoding_comparison(df, onehot, OUTPUT_DIR)
    plot_scaling_comparison(df[features], standard, minmax, OUTPUT_DIR)

    print("\n[5] ข้อมูลพร้อมเข้าโมเดลแล้ว")
    print(df_encoded.dtypes.value_counts())


if __name__ == "__main__":
    main()
