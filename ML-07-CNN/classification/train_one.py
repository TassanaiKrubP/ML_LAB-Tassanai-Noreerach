"""
train_one.py
เทรน CNN 1 การทดลอง (ถูกเรียกจาก experiments.py ใน process แยก)

    python train_one.py --run conv3

ผลที่บันทึกใน outputs/experiments/<run>/
    result.json            ตัวเลขสรุป (เป็นตัวบอกว่าการทดลองนี้เสร็จแล้ว)
    history.json           accuracy / loss ทุก epoch
    training_history.png   กราฟ train vs validation
    model.keras            โมเดล (ไม่ขึ้น git)
"""

import argparse
import gc
import json
import os
import time

import keras

from config import EXPERIMENT_DIR, OUTPUT_DIR, RANDOM_STATE, BATCH_SIZE, EPOCH_CHECKPOINTS, EARLY_STOP_PATIENCE
from cnn_model import build_model, make_dataset, make_normalization, TestAccuracyAtEpochs
from evaluate import plot_history
from experiments import get_config
from split_data import load_splits


class EpochTimer(keras.callbacks.Callback):
    """จับเวลาแต่ละ epoch"""

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    cfg = get_config(args.run)
    run_dir = os.path.join(EXPERIMENT_DIR, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    keras.utils.set_random_seed(RANDOM_STATE)   # ล็อก seed ของ python, numpy, tensorflow

    print(f"[Run] {cfg['name']}  filters={cfg['filters']}  dense={cfg['dense_units']}  "
          f"kernel={cfg['kernel_size']}  epochs={cfg['epochs']}  early_stop={cfg['early_stop']}  "
          f"batch={args.batch}", flush=True)

    with open(os.path.join(OUTPUT_DIR, "classes.json"), encoding="utf-8") as f:
        classes = json.load(f)
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    train_ds = make_dataset(X_train, y_train, shuffle=True, batch_size=args.batch)
    val_ds = make_dataset(X_val, y_val, batch_size=args.batch)
    test_ds = make_dataset(X_test, y_test, batch_size=args.batch)

    norm = make_normalization(X_train)
    model = build_model(cfg, norm, len(classes))
    params = model.count_params()
    print(f"จำนวนพารามิเตอร์ : {params:,}", flush=True)

    timer = EpochTimer()
    callbacks = [timer]
    checkpoint_cb = None
    if cfg["early_stop"]:
        callbacks += [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
    else:
        # เทรนครบทุก epoch แล้ววัด test ระหว่างทาง
        checkpoint_cb = TestAccuracyAtEpochs(test_ds, EPOCH_CHECKPOINTS)
        callbacks.append(checkpoint_cb)

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"],
                        callbacks=callbacks, verbose=2)
    train_time = time.time() - t0

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy : {test_acc:.4f}   เวลาเทรน {train_time:.0f} วินาที", flush=True)

    plot_history(hist, os.path.join(run_dir, "training_history.png"),
                 f"{cfg['name']}  (test acc {test_acc:.2%})")
    model.save(os.path.join(run_dir, "model.keras"))
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=1)

    result = {
        "name": cfg["name"], "group": cfg["group"],
        "n_conv": len(cfg["filters"]), "filters": "-".join(map(str, cfg["filters"])),
        "dense_units": cfg["dense_units"], "kernel_size": cfg["kernel_size"],
        "params": params, "batch_size": args.batch,
        "epochs_trained": len(hist["loss"]),
        "train_time_s": round(train_time, 1),
        "sec_per_epoch": round(sum(timer.times) / len(timer.times), 2),
        "best_val_accuracy": round(max(hist["val_accuracy"]), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "checkpoints": checkpoint_cb.results if checkpoint_cb else {},
    }
    # เขียน result.json เป็นอย่างสุดท้าย = ยืนยันว่าการทดลองนี้เสร็จสมบูรณ์
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
