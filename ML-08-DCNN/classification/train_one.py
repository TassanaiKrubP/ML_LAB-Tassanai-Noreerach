"""
train_one.py
เทรน DCNN 1 การทดลอง (ถูกเรียกจาก experiments.py ใน process แยก)

    python train_one.py --run vgg_small

ผลที่บันทึกใน outputs/experiments/<run>/
    result.json  history.json  training_history.png  model.keras (ไม่ขึ้น git)
"""

import argparse
import gc
import json
import os
import time

import keras

from config import (EXPERIMENT_DIR, OUTPUT_DIR, RANDOM_STATE, BATCH_SIZE, IMG_SIZE, SCRATCH_IMG_SIZE,
                    LEARNING_RATE, FINETUNE_LEARNING_RATE, EPOCH_CHECKPOINTS,
                    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE)
from vgg_model import (build_scratch, build_transfer, unfreeze_last_block, make_dataset,
                       make_normalization, count_trainable, TestAccuracyAtEpochs)
from evaluate import plot_history
from experiments import get_config, load_result
from split_data import load_splits


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.t0)


def build(cfg, X_train, n_classes):
    """สร้างโมเดลตามชนิดการทดลอง  คืน (model, เวลาเทรนก่อนหน้า, history ก่อนหน้า)"""
    if cfg["kind"] == "scratch":
        norm = make_normalization(X_train)
        return build_scratch(cfg["arch"], cfg["dense_units"], norm, n_classes, LEARNING_RATE), 0.0, None
    if cfg["kind"] == "transfer":
        return build_transfer(cfg["dense_units"], n_classes, LEARNING_RATE), 0.0, None
    # finetune : เริ่มจากโมเดล transfer ที่เทรนเสร็จแล้ว
    src = os.path.join(EXPERIMENT_DIR, "vgg16_transfer")
    model = keras.models.load_model(os.path.join(src, "model.keras"))
    with open(os.path.join(src, "history.json"), encoding="utf-8") as f:
        prev_hist = json.load(f)
    prev_time = load_result("vgg16_transfer")["train_time_s"]
    return unfreeze_last_block(model, FINETUNE_LEARNING_RATE), prev_time, prev_hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    cfg = get_config(args.run)
    run_dir = os.path.join(EXPERIMENT_DIR, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    keras.utils.set_random_seed(RANDOM_STATE)
    use_vgg_pre = cfg["kind"] in ("transfer", "finetune")

    print(f"[Run] {cfg['name']} : {cfg['description']}  dense={cfg['dense_units']}  "
          f"epochs={cfg['epochs']}  early_stop={cfg['early_stop']}  batch={args.batch}", flush=True)

    with open(os.path.join(OUTPUT_DIR, "classes.json"), encoding="utf-8") as f:
        classes = json.load(f)
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    train_ds = make_dataset(X_train, y_train, True, args.batch, use_vgg_pre)
    val_ds = make_dataset(X_val, y_val, False, args.batch, use_vgg_pre)
    test_ds = make_dataset(X_test, y_test, False, args.batch, use_vgg_pre)

    model, prev_time, prev_hist = build(cfg, X_train, len(classes))
    params, trainable = model.count_params(), count_trainable(model)
    print(f"พารามิเตอร์ทั้งหมด {params:,}  (เทรนได้ {trainable:,})", flush=True)

    timer = EpochTimer()
    callbacks = [timer]
    checkpoint_cb = None
    if cfg["early_stop"]:
        callbacks += [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=REDUCE_LR_PATIENCE, min_lr=1e-7),
        ]
    else:
        checkpoint_cb = TestAccuracyAtEpochs(test_ds, EPOCH_CHECKPOINTS)
        callbacks.append(checkpoint_cb)

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"],
                        callbacks=callbacks, verbose=2)
    train_time = time.time() - t0 + prev_time

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items() if k != "learning_rate"}
    if prev_hist:
        # ต่อกราฟของ transfer (ก่อน fine-tune) ไว้ข้างหน้า จะได้เห็นว่า fine-tune ช่วยแค่ไหน
        hist = {k: prev_hist[k] + hist[k] for k in hist}
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy : {test_acc:.4f}   เวลาเทรนรวม {train_time:.0f} วินาที", flush=True)

    plot_history(hist, os.path.join(run_dir, "training_history.png"),
                 f"{cfg['name']}  (test acc {test_acc:.2%})")
    model.save(os.path.join(run_dir, "model.keras"))
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=1)

    result = {
        "name": cfg["name"], "group": cfg["group"], "description": cfg["description"],
        "kind": cfg["kind"], "arch": cfg["arch"],
        "input_size": SCRATCH_IMG_SIZE if cfg["kind"] == "scratch" else IMG_SIZE,
        "dense_units": "-".join(map(str, cfg["dense_units"])),
        "params": params, "trainable_params": trainable, "batch_size": args.batch,
        "epochs_trained": len(timer.times),
        "train_time_s": round(train_time, 1),
        "sec_per_epoch": round(sum(timer.times) / len(timer.times), 2),
        "best_val_accuracy": round(max(history.history["val_accuracy"]), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "checkpoints": checkpoint_cb.results if checkpoint_cb else {},
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
