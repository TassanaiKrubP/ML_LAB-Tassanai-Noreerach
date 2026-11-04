"""
split_data.py
ขั้นที่ 3 : ลบรูปซ้ำ  ทำให้ทุกคลาสมีจำนวนเท่ากัน  แล้วแบ่ง train / validation / test

ทำไมต้องลบรูปซ้ำก่อนแบ่ง
    ถ้ารูปเดียวกันไปอยู่ทั้งใน train และ test โมเดลจะ "เคยเห็นข้อสอบมาก่อน"
    คะแนน test จะสูงเกินจริง (data leakage) จึงต้องลบรูปซ้ำก่อนแบ่งเสมอ
    วิธีเช็กซ้ำ : เอาพิกเซลหลังย่อรูปมาทำ hash (ลายนิ้วมือของข้อมูล) ถ้า hash ตรงกัน = รูปเดียวกัน

ทำไมต้อง 3 ส่วน
    train      โมเดลเรียนรู้ ปรับน้ำหนักจากชุดนี้เท่านั้น
    validation ลองตอบทุกจบ epoch ดูว่าเริ่มท่องจำ (overfitting) หรือยัง
    test       สอบครั้งเดียวตอนจบ เป็นคะแนนที่รายงาน
"""

import hashlib

import numpy as np
from sklearn.model_selection import train_test_split

from config import DATASET_FILE, SPLIT_FILE, MAX_PER_CLASS, TEST_SIZE, VAL_SIZE, RANDOM_STATE


def remove_duplicates(X, y):
    """เก็บรูปแรกของแต่ละ hash ไว้ ที่เหลือถือว่าซ้ำ"""
    seen, keep = set(), []
    for i, img in enumerate(X):
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return X[keep], y[keep], len(X) - len(keep)


def balance_classes(X, y, max_per_class):
    """สุ่มเลือกไม่เกิน max_per_class ใบต่อคลาส (สุ่มแบบเดิมทุกครั้งเพราะล็อก seed)"""
    rng = np.random.default_rng(RANDOM_STATE)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if max_per_class and len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.extend(sorted(idx))
    keep = np.array(keep)
    return X[keep], y[keep]


def run():
    data = np.load(DATASET_FILE)
    X, y = data["X"], data["y"]

    X, y, n_dup = remove_duplicates(X, y)
    print(f"ลบรูปซ้ำออก {n_dup} ใบ")

    X, y = balance_classes(X, y, MAX_PER_CLASS)
    print(f"หลังปรับให้คลาสเท่ากัน : {np.bincount(y).tolist()} ใบต่อคลาส")

    # รอบแรก แยก test ออกไปก่อน
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # รอบสอง แยก validation ออกจากส่วนที่เหลือ (ปรับสัดส่วนเพราะฐานเหลือไม่ถึง 100%)
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_rest
    )

    np.savez_compressed(SPLIT_FILE, X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    total = len(y)
    print(f"train      {len(y_train):>5} ใบ  ({len(y_train)/total:.0%})")
    print(f"validation {len(y_val):>5} ใบ  ({len(y_val)/total:.0%})")
    print(f"test       {len(y_test):>5} ใบ  ({len(y_test)/total:.0%})")
    return {"duplicates_removed": n_dup, "per_class": np.bincount(y).tolist(),
            "train": len(y_train), "val": len(y_val), "test": len(y_test)}


def load_splits():
    """โหลดชุดที่แบ่งไว้แล้ว (ทุกไฟล์เรียกใช้ฟังก์ชันนี้ ให้ได้ชุด test เดียวกันเสมอ)"""
    d = np.load(SPLIT_FILE)
    return (d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"])


if __name__ == "__main__":
    run()
