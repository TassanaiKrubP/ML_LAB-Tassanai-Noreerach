"""
split_data.py
ขั้นที่ 2 : แบ่งข้อมูลเป็นชุดสอน (train) และชุดสอบ (test)

ทำไมต้องแบ่งก่อน Standardize ?
    เพราะถ้าคำนวณค่าเฉลี่ยจากข้อมูลทั้งก้อน ข้อมูลชุดสอบจะแอบรั่วเข้าไปในขั้นเตรียมข้อมูล
    เหมือนให้นักเรียนแอบเห็นข้อสอบก่อนสอบ คะแนนที่ได้จะสูงเกินจริง
"""

import numpy as np
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, TEST_SIZE, RANDOM_STATE


def run():
    X = np.load(OUTPUT_DIR / "features.npy")
    y = np.load(OUTPUT_DIR / "labels.npy")

    # stratify=y ทำให้สัดส่วนของแต่ละคลาสในชุดสอนและชุดสอบเท่ากัน
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    print(f"ชุดสอน {X_train.shape[0]} ใบ   ชุดสอบ {X_test.shape[0]} ใบ")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run()
