"""
Part 2 - Data Visualization

ตัวเลขใน describe() บอกได้แค่ว่า "ค่าเฉลี่ยเท่าไหร่" แต่กราฟบอกได้ว่า
"ข้อมูลกระจายตัวหน้าตาแบบไหน" ซึ่งช่วยให้เห็นค่าผิดปกติได้เร็วกว่ามาก
"""

import pandas as pd

from load_data import load_data, OUTPUT_DIR
from plot_result import plot_histograms, plot_correlation_heatmap, plot_boxplot


def main():
    print("=" * 55)
    print("02 - Data Visualization")
    print("=" * 55)

    df = load_data()

    # career_points ยังเป็นข้อความอยู่ ต้องแปลงก่อนถึงจะเอามาคำนวณได้
    df["career_points"] = pd.to_numeric(
        df["career_points"].astype(str).str.replace(",", ""), errors="coerce"
    )

    numeric = df.select_dtypes(include="number")
    print(f"\nคอลัมน์ตัวเลขที่จะเอามาวาดกราฟ: {list(numeric.columns)}")

    print("\n[1] Histogram")
    plot_histograms(numeric, OUTPUT_DIR)

    print("\n[2] Correlation Heatmap")
    corr = numeric.corr()
    print(corr.round(2))
    plot_correlation_heatmap(corr, OUTPUT_DIR)

    print("\nคู่ที่สัมพันธ์กันสูงสุด (ไม่นับตัวเอง)")
    pairs = corr.abs().unstack().sort_values(ascending=False)
    pairs = pairs[pairs < 0.999]
    print(pairs.head(6))

    print("\n[3] Boxplot (ดูค่าผิดปกติ)")
    plot_boxplot(df, OUTPUT_DIR)
    print("จุดที่หลุดออกไปไกลจากกล่องคือค่าที่น่าสงสัย เดี๋ยวไปแก้ใน Part 3")


if __name__ == "__main__":
    main()
