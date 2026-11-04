"""
experiments.py
ขั้นที่ 5 : กำหนดชุดการทดลองทั้งหมด แล้วสั่ง train_one.py ทีละการทดลองใน process แยก

ทำไมแยก process
    เครื่องมี RAM 8 GB ถ้าเทรนหลายโมเดลต่อกันใน process เดียว หน่วยความจำจะค้างสะสม
    แยก process ทำให้ RAM คืนหมดทุกครั้งที่จบการทดลอง

เทรนต่อจากเดิมได้ (resume)
    การทดลองที่มี result.json แล้วจะถูกข้าม ถ้าเครื่องดับกลางคืน รัน main.py ใหม่ก็ทำต่อได้เลย
"""

import csv
import json
import os
import subprocess
import sys

from config import BASE_DIR, EXPERIMENT_DIR, LOG_DIR, OUTPUT_DIR, EPOCHS, EPOCH_RUN_EPOCHS, BATCH_SIZE

BASELINE = {"filters": [32, 64, 128], "dense_units": 128, "kernel_size": 3}

# name -> ค่าที่ต่างจาก baseline + กลุ่มที่ใช้เปรียบเทียบ
RUNS = {
    # A. เทียบจำนวน epoch : เทรนยาว 30 epoch ไม่หยุดก่อน วัด test ที่ 5/10/15/20/30
    "epochs_30":  {"group": "epochs", "epochs": EPOCH_RUN_EPOCHS, "early_stop": False},
    # B. เทียบจำนวนชั้น convolution
    "conv1": {"group": "conv_layers", "filters": [32]},
    "conv2": {"group": "conv_layers", "filters": [32, 64]},
    "conv3": {"group": "conv_layers", "filters": [32, 64, 128]},          # = baseline
    "conv4": {"group": "conv_layers", "filters": [32, 64, 128, 256]},
    # C. เทียบจำนวนนิวรอนใน Dense (dense128 คือ conv3 ใช้ผลร่วมกัน)
    "dense64":  {"group": "neurons", "dense_units": 64},
    "dense256": {"group": "neurons", "dense_units": 256},
    # D. (โบนัส) ขนาด kernel 5x5 เทียบกับ 3x3 (= conv3)
    "kernel5": {"group": "kernel", "kernel_size": 5},
}


def get_config(name):
    cfg = dict(BASELINE)
    cfg.update({"epochs": EPOCHS, "early_stop": True})
    cfg.update(RUNS[name])
    cfg["name"] = name
    return cfg


def result_path(name):
    return os.path.join(EXPERIMENT_DIR, name, "result.json")


def run_one(name, batch_size=BATCH_SIZE):
    """รัน train_one.py ใน process ใหม่ พร้อมเขียน log ลงไฟล์และแสดงบนจอไปพร้อมกัน"""
    if os.path.exists(result_path(name)):
        print(f"  ข้าม {name} (มีผลแล้ว)")
        return True

    log_file = os.path.join(LOG_DIR, f"{name}.log")
    cmd = [sys.executable, "-u", os.path.join(BASE_DIR, "train_one.py"),
           "--run", name, "--batch", str(batch_size)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="2")
    with open(log_file, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env, cwd=BASE_DIR)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        proc.wait()

    if proc.returncode != 0:
        print(f"  !! {name} ล้มเหลว (exit {proc.returncode}) ดู {log_file}")
        # ส่วนใหญ่ล้มเพราะ RAM ไม่พอ ลองใหม่ด้วย batch ครึ่งหนึ่ง 1 ครั้ง
        if batch_size > 4:
            print(f"  ลอง {name} ใหม่ด้วย batch {batch_size // 2}")
            return run_one(name, batch_size // 2)
        return False
    return True


def collect_csv():
    """รวม result.json ทุกการทดลองเป็น outputs/experiments.csv"""
    rows = []
    for name in RUNS:
        p = result_path(name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            rows.append({k: r[k] for k in ["name", "group", "n_conv", "filters", "dense_units",
                                           "kernel_size", "params", "epochs_trained",
                                           "train_time_s", "sec_per_epoch",
                                           "best_val_accuracy", "test_accuracy"]})
    path = os.path.join(OUTPUT_DIR, "experiments.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"บันทึก {path} ({len(rows)} การทดลอง)")
    return rows


def run_all():
    failed = [name for name in RUNS if not run_one(name)]
    if failed:
        print(f"การทดลองที่ล้มเหลว : {failed}")
    return collect_csv()


if __name__ == "__main__":
    run_all()
