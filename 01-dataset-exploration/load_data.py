import os

import pandas as pd

# f1_drivers_raw.csv อยู่สูงขึ้นไปหนึ่งระดับ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "f1_drivers_raw.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    """อ่านข้อมูลดิบนักแข่ง F1 (ยังไม่ทำความสะอาด)"""

    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    return df
