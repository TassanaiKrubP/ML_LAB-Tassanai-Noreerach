"""
main.py
Lab 8 : DCNN (VGG) จำแนกรูป Airplane / Bird / Drone  (รันไฟล์นี้ไฟล์เดียวจบทุกขั้น)

    python main.py

ขั้นตอน
    1  โหลดรูป ข้ามไฟล์เสีย ย่อเป็น 96x96 RGB (uint8)
    2  ลบรูปซ้ำ แบ่ง train / val / test (เหมือน Lab 7 ทุกอย่าง)
    3  เทรนทุกการทดลอง (สถาปัตยกรรม, นิวรอน, epoch) ทีละ process
    4  เลือกโมเดลที่ดีที่สุดจาก validation accuracy แล้ววัดผลบน test
    5  วาดกราฟเปรียบเทียบ รวมถึงกราฟเทียบทุกแล็ป
    6  ทายรูปสุ่ม 4 ใบ

ถ้าหยุดกลางทาง รันใหม่จะทำต่อจากจุดเดิม
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
    """เลือกจาก validation accuracy (ไม่ใช้ test เลือก) ถ้าเท่ากันเลือกตัวที่พารามิเตอร์น้อยกว่า
    ไม่นับการทดลอง epoch เพราะไม่มี EarlyStopping โมเดลที่บันทึกคือ epoch สุดท้าย
    ไม่ใช่ epoch ที่ val accuracy สูงสุด"""
    candidates = [r for r in rows if r["group"] != "epochs"]
    best = max(candidates, key=lambda r: (r["best_val_accuracy"], -r["params"]))
    with open(os.path.join(OUTPUT_DIR, "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best


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

    print("\n[Step 4] Evaluating the best model on the test set...")
    best = pick_best(rows)
    print(f"  โมเดลที่ดีที่สุด (val accuracy สูงสุด) : {best['name']}")

    import keras
    import evaluate
    from vgg_model import make_dataset
    import test_vgg

    with open(os.path.join(OUTPUT_DIR, "classes.json"), encoding="utf-8") as f:
        classes = json.load(f)
    with open(os.path.join(EXPERIMENT_DIR, best["name"], "result.json"), encoding="utf-8") as f:
        kind = json.load(f)["kind"]
    _, _, _, _, X_test, y_test = split_data.load_splits()
    model = keras.models.load_model(os.path.join(EXPERIMENT_DIR, best["name"], "model.keras"))
    test_ds = make_dataset(X_test, y_test, vgg_preprocess=kind in ("transfer", "finetune"))
    evaluate.evaluate_model(model, test_ds, y_test, classes, OUTPUT_DIR,
                            title=f"Best DCNN: {best['name']}")

    print("\n[Step 5] Plotting comparison charts...")
    plot_result.run(best)

    print("\n[Step 6] Predicting 4 random test images...")
    test_vgg.run(best["name"])

    print(f"\nเสร็จทั้งหมดใน {(time.time() - t_start) / 60:.1f} นาที")


if __name__ == "__main__":
    main()
