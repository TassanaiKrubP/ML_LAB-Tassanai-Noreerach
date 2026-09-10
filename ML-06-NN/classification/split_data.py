"""
split_data.py
ขั้นที่ 3 : แบ่งข้อมูลเป็น 3 ส่วน train / validation / test

ทำไมต้อง 3 ส่วน
    train      โมเดลใช้เรียนรู้ ปรับน้ำหนักจากชุดนี้เท่านั้น
    validation โมเดลไม่ได้เรียนจากชุดนี้ แต่เอามาลองตอบทุกจบ epoch
               ใช้ดูว่ายังเก่งขึ้นอยู่ หรือเริ่มท่องจำข้อมูลสอนแล้ว
    test       เก็บไว้สอบครั้งเดียวตอนจบ เป็นคะแนนที่รายงานได้อย่างซื่อสัตย์

    ถ้าใช้ validation ตัดสินใจปรับโมเดลไปเรื่อย ๆ สุดท้ายเราจะเผลอปรับจนเข้ากับ
    validation โดยไม่รู้ตัว จึงต้องมี test ที่ไม่เคยถูกใช้เลยไว้เป็นกรรมการกลาง
"""

import numpy as np
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, TEST_SIZE, VAL_SIZE, RANDOM_STATE


def run():
    X = np.load(OUTPUT_DIR / "features.npy")
    y = np.load(OUTPUT_DIR / "labels.npy")

    # รอบแรก แยก test ออกไปก่อน
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # รอบสอง แยก validation ออกจากส่วนที่เหลือ
    # ต้องปรับสัดส่วนใหม่ เพราะตอนนี้ฐานเหลือไม่ถึง 100% แล้ว
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_ratio,
        random_state=RANDOM_STATE, stratify=y_rest
    )

    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                      ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        np.save(OUTPUT_DIR / f"{name}.npy", arr)

    total = len(y)
    print(f"train      {len(y_train):>5} ใบ  ({len(y_train)/total:.0%})")
    print(f"validation {len(y_val):>5} ใบ  ({len(y_val)/total:.0%})")
    print(f"test       {len(y_test):>5} ใบ  ({len(y_test)/total:.0%})")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    run()
