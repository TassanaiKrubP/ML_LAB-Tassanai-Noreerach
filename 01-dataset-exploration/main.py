"""
Part 1 - Dataset Exploration

สำรวจข้อมูลก่อนแตะต้องอะไรทั้งสิ้น เหมือนเปิดตู้เย็นดูของก่อนจะเริ่มทำอาหาร
ว่ามีวัตถุดิบอะไรบ้าง อันไหนหมดอายุ อันไหนซื้อซ้ำมาสองอัน
"""

from load_data import load_data, OUTPUT_DIR
from plot_result import plot_class_distribution


def main():
    print("=" * 55)
    print("01 - Dataset Exploration")
    print("=" * 55)

    df = load_data()

    print("\n[1] Display Shape")
    print(f"ข้อมูลมี {df.shape[0]} แถว และ {df.shape[1]} คอลัมน์")

    print("\n[2] Display Data Types")
    print(df.dtypes)
    print("\nสังเกต career_points เป็นข้อความ (object/str) ทั้งที่ควรเป็นตัวเลข")

    print("\n[3] Display Head")
    print(df.head())

    print("\n[4] Display Summary Statistics")
    print(df.describe())
    print("\nสังเกต age มีค่า min ติดลบ และ max = 999 ซึ่งเป็นไปไม่ได้")

    print("\n[5] Display Missing Values")
    missing = df.isna().sum()
    print(missing[missing > 0])
    print(f"\nรวมค่าที่หายไปทั้งหมด {int(missing.sum())} ช่อง")

    print("\n[6] Display Duplicate Records")
    duplicated = df[df.duplicated(keep=False)]
    print(f"พบแถวซ้ำ {df.duplicated().sum()} แถว")
    print(duplicated[["name", "team", "age"]].sort_values("name"))

    print("\n[7] Display Class Distribution")
    # ใช้ team เป็น class เพราะเป็นคอลัมน์ประเภทข้อความที่จะเอาไป encode ใน Part 4
    counts = df["team"].value_counts(dropna=False)
    print(counts)
    print("\nสังเกตว่า Ferrari ถูกนับแยกกันหลายอัน เพราะพิมพ์ไม่เหมือนกัน")

    print("\n[8] วาดกราฟ")
    plot_class_distribution(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
