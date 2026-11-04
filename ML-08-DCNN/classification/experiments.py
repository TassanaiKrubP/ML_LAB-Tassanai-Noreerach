"""
experiments.py
ขั้นที่ 5 : กำหนดชุดการทดลอง แล้วสั่ง train_one.py ทีละการทดลองใน process แยก (RAM คืนหมดทุกครั้ง)

ลำดับการทดลอง
    A  เทียบสถาปัตยกรรม   vgg_small / vgg16_scratch / vgg16_transfer / vgg16_finetune
    B  เทียบจำนวนนิวรอนในหัว Dense ของ VGG ที่เทรนเองตัวที่ดีกว่าจาก A
    C  เทียบจำนวน epoch ของโมเดลที่ดีที่สุดจาก A+B (เทรน 30 epoch ไม่หยุดก่อน)
ตัวที่เลือกจาก A / B ถูกจดใน outputs/choices.json เพื่อให้รันต่อจากเดิมได้ผลเหมือนเดิม
"""

import csv
import json
import os
import subprocess
import sys

from config import (BASE_DIR, EXPERIMENT_DIR, LOG_DIR, OUTPUT_DIR, EPOCHS, EPOCH_RUN_EPOCHS,
                    FINETUNE_EPOCHS, BATCH_SIZE)

CHOICES_FILE = os.path.join(OUTPUT_DIR, "choices.json")

ARCH_RUNS = ["vgg_small", "vgg16_scratch", "vgg16_transfer", "vgg16_finetune"]
NEURON_RUNS = ["neurons_256_128", "neurons_128_64"]
EPOCH_RUN = "epochs_30"
ALL_RUNS = ARCH_RUNS + NEURON_RUNS + [EPOCH_RUN]

FIXED = {
    "vgg_small":      {"group": "architecture", "kind": "scratch", "arch": "vgg_small",
                       "description": "VGG_SMALL from scratch (10 conv)"},
    "vgg16_scratch":  {"group": "architecture", "kind": "scratch", "arch": "vgg16",
                       "description": "VGG16 layout from scratch (13 conv)"},
    "vgg16_transfer": {"group": "architecture", "kind": "transfer", "arch": "vgg16",
                       "description": "VGG16 ImageNet, frozen + new head"},
    "vgg16_finetune": {"group": "architecture", "kind": "finetune", "arch": "vgg16",
                       "epochs": FINETUNE_EPOCHS,
                       "description": "transfer + unfreeze block5, LR 1e-5, +5 epochs"},
}


def load_choices():
    if os.path.exists(CHOICES_FILE):
        with open(CHOICES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_choice(key, value):
    choices = load_choices()
    if key not in choices:          # เลือกครั้งเดียว รันซ้ำไม่เปลี่ยน
        choices[key] = value
        with open(CHOICES_FILE, "w", encoding="utf-8") as f:
            json.dump(choices, f, indent=2)
    return load_choices()[key]


def get_config(name):
    cfg = {"dense_units": [512, 256], "epochs": EPOCHS, "early_stop": True}
    if name in FIXED:
        cfg.update(FIXED[name])
    elif name in NEURON_RUNS:
        base = load_choices()["best_scratch"]
        units = [int(u) for u in name.split("_")[1:]]
        cfg.update(FIXED[base])
        cfg.update({"group": "neurons", "dense_units": units,
                    "description": f"{base} with dense {units[0]}-{units[1]}"})
    elif name == EPOCH_RUN:
        base = load_choices()["best_for_epochs"]
        cfg = get_config(base)
        cfg.update({"group": "epochs", "epochs": EPOCH_RUN_EPOCHS, "early_stop": False,
                    "description": f"{base} trained 30 epochs, no early stopping"})
    cfg["name"] = name
    return cfg


def result_path(name):
    return os.path.join(EXPERIMENT_DIR, name, "result.json")


def load_result(name):
    with open(result_path(name), encoding="utf-8") as f:
        return json.load(f)


def run_one(name, batch_size=BATCH_SIZE):
    """รัน train_one.py ใน process ใหม่ เขียน log ลงไฟล์และแสดงบนจอไปพร้อมกัน"""
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
        if batch_size > 4:
            print(f"  ลอง {name} ใหม่ด้วย batch {batch_size // 2}")
            return run_one(name, batch_size // 2)
        return False
    return True


def best_by_val(names):
    done = [load_result(n) for n in names if os.path.exists(result_path(n))]
    return max(done, key=lambda r: (r["best_val_accuracy"], -r["params"]))["name"]


def collect_csv():
    keys = ["name", "group", "description", "input_size", "dense_units", "params",
            "trainable_params", "batch_size", "epochs_trained", "train_time_s", "sec_per_epoch",
            "best_val_accuracy", "test_accuracy"]
    rows = [{k: load_result(n)[k] for k in keys} for n in ALL_RUNS if os.path.exists(result_path(n))]
    path = os.path.join(OUTPUT_DIR, "experiments.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"บันทึก {path} ({len(rows)} การทดลอง)")
    return rows


def run_all():
    print("  -- A. architecture --")
    for name in ARCH_RUNS:
        run_one(name)

    best_scratch = save_choice("best_scratch", best_by_val(["vgg_small", "vgg16_scratch"]))
    print(f"  VGG เทรนเองที่ดีกว่า : {best_scratch}  -> ใช้ทดลองจำนวนนิวรอน")
    print("  -- B. neurons --")
    for name in NEURON_RUNS:
        run_one(name)

    # finetune ไม่ใช้ในการทดลอง epoch เพราะต้องเทรนต่อจากโมเดลอื่น
    best = save_choice("best_for_epochs", best_by_val(
        ["vgg_small", "vgg16_scratch", "vgg16_transfer"] + NEURON_RUNS))
    print(f"  โมเดลที่ใช้ทดลองจำนวน epoch : {best}")
    print("  -- C. epochs --")
    run_one(EPOCH_RUN)
    return collect_csv()


if __name__ == "__main__":
    run_all()
