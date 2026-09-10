"""
svm_model.py
ขั้นที่ 4 : เทรน SVM ด้วย 3 kernel แล้วเปรียบเทียบ accuracy

kernel คืออะไร
    linear : ลากเส้นตรงแบ่งสองฝั่ง  เร็วที่สุด เหมาะกับข้อมูลที่แยกกันตรงไปตรงมา
    poly   : ลากเส้นโค้งแบบสมการยกกำลัง  ยืดหยุ่นขึ้น แต่ช้าและปรับยาก
    rbf    : ลากขอบเขตโค้งอิสระได้ทุกทิศ  มักได้ผลดีที่สุดกับรูปภาพ
"""

import time

import joblib
import numpy as np
from sklearn.svm import SVC

from config import OUTPUT_DIR, KERNELS, RANDOM_STATE, MAX_ITER, CACHE_SIZE


def run():
    X_train = np.load(OUTPUT_DIR / "X_train.npy")
    X_test = np.load(OUTPUT_DIR / "X_test.npy")
    y_train = np.load(OUTPUT_DIR / "y_train.npy")
    y_test = np.load(OUTPUT_DIR / "y_test.npy")

    results = {}
    best_model, best_kernel, best_score = None, None, -1.0

    print(f"\n{'kernel':<12}{'accuracy':>12}{'เวลาเทรน (วินาที)':>22}")
    print("-" * 46)

    for kernel in KERNELS:
        model = SVC(kernel=kernel, random_state=RANDOM_STATE,
            max_iter=MAX_ITER, cache_size=CACHE_SIZE)

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        score = model.score(X_test, y_test)
        results[kernel] = score
        print(f"{kernel:<12}{score:>11.4f}{elapsed:>20.2f}")

        if score > best_score:
            best_model, best_kernel, best_score = model, kernel, score

    joblib.dump(best_model, OUTPUT_DIR / "svm_model.pkl")
    print("-" * 46)
    print(f"kernel ที่ดีที่สุดคือ {best_kernel}  accuracy {best_score:.4f}")
    print(f"บันทึกโมเดลลง {OUTPUT_DIR / 'svm_model.pkl'}")

    return results, best_kernel


if __name__ == "__main__":
    run()
