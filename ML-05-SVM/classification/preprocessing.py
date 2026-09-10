"""
preprocessing.py
ขั้นที่ 3 : Standardize feature ตามที่ใบงานกำหนด แล้วบีบมิติด้วย PCA

StandardScaler ทำอะไร
    ปรับทุก feature ให้มีค่าเฉลี่ย 0 และการกระจาย 1
    SVM ตัดสินใจจาก "ระยะห่าง" ถ้า feature ตัวหนึ่งมีตัวเลขใหญ่กว่าตัวอื่นมาก
    มันจะครอบงำการคำนวณระยะทางไปหมด การปรับสเกลทำให้ทุก feature มีสิทธิ์เท่ากัน

PCA ทำอะไร
    บีบ 4096 feature เหลือ 100 feature โดยเก็บข้อมูลสำคัญไว้ให้มากที่สุด
    ทำให้ SVM เทรนเร็วขึ้นหลายเท่า และลดอาการโมเดลจำข้อมูลสอน (overfitting)

สำคัญ : fit ด้วยชุดสอนเท่านั้น แล้วนำสูตรเดิมไป transform ชุดสอบ
"""

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR, PCA_COMPONENTS, RANDOM_STATE


def build_pipeline(n_features):
    steps = [("scaler", StandardScaler())]
    if PCA_COMPONENTS:
        # จำนวนมิติปลายทางต้องไม่เกินจำนวน feature ที่มีอยู่จริง
        n_components = min(PCA_COMPONENTS, n_features)
        steps.append(("pca", PCA(n_components=n_components,
                                 random_state=RANDOM_STATE)))
    return Pipeline(steps)


def run():
    X_train = np.load(OUTPUT_DIR / "X_train.npy")
    X_test = np.load(OUTPUT_DIR / "X_test.npy")

    pipeline = build_pipeline(X_train.shape[1])

    X_train_s = pipeline.fit_transform(X_train)   # fit + transform เฉพาะชุดสอน
    X_test_s = pipeline.transform(X_test)         # ชุดสอบใช้แค่ transform

    # เขียนทับด้วยเวอร์ชันที่ปรับสเกลแล้ว เพื่อให้ขั้นถัดไปหยิบไปใช้ได้เลย
    np.save(OUTPUT_DIR / "X_train.npy", X_train_s)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_s)
    joblib.dump(pipeline, OUTPUT_DIR / "scaler.pkl")

    print(f"ปรับสเกลเรียบร้อย  มิติข้อมูล {X_train.shape[1]} -> {X_train_s.shape[1]}")
    return X_train_s, X_test_s


if __name__ == "__main__":
    run()
