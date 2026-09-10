"""
nn_model.py
ขั้นที่ 4 : สร้าง เทรน และเปรียบเทียบ Neural Network หลายโครงสร้าง

โครงสร้างของโมเดล
    Flatten   คลี่รูป 64x64x3 ให้เป็นแถวตัวเลขยาว 12,288 ตัว
    Dense     ชั้นซ่อน แต่ละนิวรอนรับค่าจากทุกตัวในชั้นก่อนหน้า
    Dropout   สุ่มปิดนิวรอนบางส่วนระหว่างเทรน กันไม่ให้โมเดลท่องจำ
    Dense     ชั้นสุดท้าย จำนวนนิวรอนเท่าจำนวนคลาส softmax แปลงเป็นความน่าจะเป็น

เรื่อง epoch ที่ใบงานให้เปรียบเทียบ
    1 epoch = โมเดลดูข้อมูลชุดสอน "กองเดิม" ครบทั้งกอง 1 รอบ ไม่ใช่ข้อมูลใหม่
    โค้ดนี้เทรนยาวครั้งเดียวจนครบ EPOCHS แล้วอ่านค่าย้อนหลังจากประวัติการเทรน
    ได้ผลเท่ากับการเทรนแยกหลายรอบ แต่ใช้เวลาน้อยกว่ามาก
"""

import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (OUTPUT_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE,
                    CONFIGURATIONS, EPOCH_CHECKPOINTS, RANDOM_STATE)


def build_model(input_shape, n_classes, hidden, dropout):
    """ประกอบโมเดลตามรายการจำนวนนิวรอนที่ส่งเข้ามา"""
    layers = [keras.layers.Input(shape=input_shape), keras.layers.Flatten()]

    for units in hidden:
        layers.append(keras.layers.Dense(units, activation="relu"))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))

    layers.append(keras.layers.Dense(n_classes, activation="softmax"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run():
    keras.utils.set_random_seed(RANDOM_STATE)

    X_train = np.load(OUTPUT_DIR / "X_train.npy")
    X_val = np.load(OUTPUT_DIR / "X_val.npy")
    y_train = np.load(OUTPUT_DIR / "y_train.npy")
    y_val = np.load(OUTPUT_DIR / "y_val.npy")

    with open(OUTPUT_DIR / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    input_shape = X_train.shape[1:]
    all_history = {}
    best_model, best_name, best_score = None, None, -1.0

    for cfg in CONFIGURATIONS:
        print(f"\n{'-' * 60}")
        print(f"โครงสร้าง : {cfg['name']}   ชั้นซ่อน {cfg['hidden']}   dropout {cfg['dropout']}")
        print("-" * 60)

        model = build_model(input_shape, len(classes), cfg["hidden"], cfg["dropout"])
        print(f"จำนวนพารามิเตอร์ที่ต้องเรียนรู้ {model.count_params():,} ตัว")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )

        all_history[cfg["name"]] = {k: [float(v) for v in vals]
                                    for k, vals in history.history.items()}

        final_val = history.history["val_accuracy"][-1]
        if final_val > best_score:
            best_model, best_name, best_score = model, cfg["name"], final_val

    # ---------- ตารางเปรียบเทียบตามจำนวน epoch ----------
    print(f"\n{'=' * 70}")
    print("เปรียบเทียบ val_accuracy ที่จำนวน epoch ต่าง ๆ")
    print("=" * 70)
    header = f"{'โครงสร้าง':<24}" + "".join(f"{f'ep {e}':>9}" for e in EPOCH_CHECKPOINTS)
    print(header)
    print("-" * 70)
    for name, hist in all_history.items():
        row = f"{name:<24}"
        for e in EPOCH_CHECKPOINTS:
            row += f"{hist['val_accuracy'][e - 1]:>9.4f}" if e <= EPOCHS else f"{'-':>9}"
        print(row)

    best_model.save(OUTPUT_DIR / "nn_model.keras")
    with open(OUTPUT_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(all_history, f, ensure_ascii=False, indent=2)

    print(f"\nโครงสร้างที่ดีที่สุดคือ {best_name}  val_accuracy {best_score:.4f}")
    print(f"บันทึกโมเดลลง {OUTPUT_DIR / 'nn_model.keras'}")

    return all_history, best_name


if __name__ == "__main__":
    run()
