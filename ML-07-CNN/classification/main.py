"""
main.py
Lab 7 : CNN จำแนกรูป Airplane / Bird / Drone  (รันไฟล์นี้ไฟล์เดียวจบทุกขั้น)

    python main.py

ขั้นตอน
    1  โหลดรูป ข้ามไฟล์เสีย ย่อเป็น 96x96 RGB (uint8)
    2  ลบรูปซ้ำ ปรับคลาสให้เท่ากัน แบ่ง train / val / test
    3  เทรนทุกการทดลอง (epoch, จำนวนชั้น conv, นิวรอน, kernel) ทีละ process
    4  วาดกราฟเปรียบเทียบ
    5  เลือกโมเดลที่ดีที่สุดจาก validation accuracy แล้ววัดผลบน test
    6  วาด feature map และ kernel ของชั้น conv
    7  ทายรูปสุ่ม 4 ใบ

ถ้าหยุดกลางทาง รันใหม่จะทำต่อจากจุดเดิม (ขั้นที่เสร็จแล้วถูกข้าม)
"""

import json
import os
import time

from config import DATASET_FILE, SPLIT_FILE, OUTPUT_DIR, EXPERIMENT_DIR

import preprocessing
import split_data
import experiments
import plot_result


def pick_best(rows):
    """เลือกจาก validation accuracy (ไม่ใช้ test เลือก เพื่อให้คะแนน test ยังเป็นกลาง)
    ถ้าเท่ากัน เลือกตัวที่พารามิเตอร์น้อยกว่า
    ไม่นับการทดลอง epoch เพราะไม่มี EarlyStopping โมเดลที่บันทึกคือ epoch สุดท้าย
    ไม่ใช่ epoch ที่ val accuracy สูงสุด"""
    candidates = [r for r in rows if r["group"] != "epochs"]
    best = max(candidates, key=lambda r: (r["best_val_accuracy"], -r["params"]))
    with open(os.path.join(OUTPUT_DIR, "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best["name"]


def main():
    t_start = time.time()

    print("[Step 1] Loading and preprocessing images...")
    if not os.path.exists(DATASET_FILE):
        preprocessing.run()
    else:
        print("  มี dataset.npz แล้ว ข้าม")

    print("\n[Step 2] Removing duplicates and splitting train / val / test...")
    if not os.path.exists(SPLIT_FILE):
        info = split_data.run()
        with open(os.path.join(OUTPUT_DIR, "split_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    else:
        print("  มี splits.npz แล้ว ข้าม")

    print("\n[Step 3] Training all experiments (one subprocess per run)...")
    rows = experiments.run_all()

    print("\n[Step 4] Plotting comparison charts...")
    plot_result.run()

    print("\n[Step 5] Evaluating the best model on the test set...")
    best = pick_best(rows)
    print(f"  โมเดลที่ดีที่สุด (val accuracy สูงสุด) : {best}")

    # import ตรงนี้ เพราะขั้นก่อนหน้าไม่ต้องโหลด TensorFlow ใน process หลัก
    import keras
    import evaluate
    from cnn_model import make_dataset
    import visualize_features
    import test_cnn

    with open(os.path.join(OUTPUT_DIR, "classes.json"), encoding="utf-8") as f:
        classes = json.load(f)
    _, _, _, _, X_test, y_test = split_data.load_splits()
    model = keras.models.load_model(os.path.join(EXPERIMENT_DIR, best, "model.keras"))
    evaluate.evaluate_model(model, make_dataset(X_test, y_test), y_test, classes, OUTPUT_DIR,
                            title=f"Best CNN: {best}")

    print("\n[Step 6] Visualizing feature maps...")
    visualize_features.run(best)

    print("\n[Step 7] Predicting 4 random test images...")
    test_cnn.run(best)

    print(f"\nเสร็จทั้งหมดใน {(time.time() - t_start) / 60:.1f} นาที")


if __name__ == "__main__":
    main()
