"""
04-evaluate/main.py
ขั้นที่ 4 : วัดผลทุกการทดลองบนชุด test (ภาพที่โมเดลไม่เคยเห็นทั้งตอนเทรนและตอนเลือกโมเดล)

    Precision   กรอบที่โมเดลทายมา ถูกกี่ %          (ทายมั่ว = precision ต่ำ)
    Recall      หน้าคนที่มีจริง โมเดลหาเจอกี่ %       (หาไม่เจอ = recall ต่ำ)
    mAP50       ค่าเฉลี่ย AP ทุกคลาส นับว่าถูกเมื่อกรอบซ้อนกรอบจริง IoU >= 0.5
    mAP50-95    เฉลี่ย mAP ที่ IoU 0.50, 0.55, ..., 0.95  เข้มกว่า ต้องวางกรอบแม่นจริง

ผลลัพธ์
    outputs/metrics.csv       ตารางทุกการทดลอง
    outputs/best_run.json     การทดลองที่ดีที่สุด (เลือกจาก val mAP50-95 ไม่ใช้ test เลือก)
    outputs/results.png, confusion_matrix.png, PR_curve.png ของการทดลองที่ดีที่สุด
"""

import csv
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultralytics import YOLO

from common import RUNS, ALIASES, CLASSES, DATA_YAML, TRAIN_DIR, EVAL_DIR, BEST_RUN_FILE, best_weights

VAL_DIR = os.path.join(EVAL_DIR, "test_runs")


def read_results_csv(run_name):
    """อ่าน results.csv ของ ultralytics (ค่าทุก epoch)  ตัดช่องว่างในชื่อคอลัมน์"""
    with open(os.path.join(TRAIN_DIR, run_name, "results.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): float(v) for k, v in r.items()} for r in rows]


def evaluate_run(name):
    cfg = RUNS[name]
    model = YOLO(best_weights(name))
    m = model.val(data=DATA_YAML, split="test", imgsz=cfg["imgsz"], batch=16, device=0,
                  workers=0, project=VAL_DIR, name=name, exist_ok=True, plots=True, verbose=False)

    with open(os.path.join(TRAIN_DIR, name, "result.json"), encoding="utf-8") as f:
        train_info = json.load(f)
    history = read_results_csv(name)

    row = {"run": name, "group": cfg["group"], "epochs": cfg["epochs"], "imgsz": cfg["imgsz"],
           "batch": train_info["batch"],
           "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
           "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)}
    ap50 = dict(zip([int(i) for i in m.box.ap_class_index], m.box.ap50))
    for i, c in enumerate(CLASSES):
        row[f"AP50_{c}"] = round(float(ap50.get(i, 0.0)), 4)
    row["best_val_mAP50-95"] = round(max(h["metrics/mAP50-95(B)"] for h in history), 4)
    row["train_time_min"] = round(train_info["train_time_s"] / 60, 1)
    row["alias_of"] = ""
    return row


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    print("[Step 4] Evaluating every run on the TEST split...")
    rows = []
    for name in RUNS:
        if not os.path.exists(best_weights(name)):
            print(f"  ข้าม {name} (ยังไม่มี best.pt)")
            continue
        row = evaluate_run(name)
        rows.append(row)
        print(f"  {name:<9} P={row['precision']:.3f}  R={row['recall']:.3f}  "
              f"mAP50={row['mAP50']:.3f}  mAP50-95={row['mAP50-95']:.3f}")

    # s640_b16 ตั้งค่าเหมือน e30 ทุกอย่าง ใช้ผลเดียวกัน
    for alias, source in ALIASES.items():
        src = next((r for r in rows if r["run"] == source), None)
        if src:
            rows.append({**src, "run": alias, "group": "config", "alias_of": source})

    with open(os.path.join(EVAL_DIR, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("  บันทึก outputs/metrics.csv")

    best = max((r for r in rows if not r["alias_of"]), key=lambda r: r["best_val_mAP50-95"])
    with open(BEST_RUN_FILE, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"  การทดลองที่ดีที่สุด (val mAP50-95 สูงสุด) : {best['run']}")

    # คัดลอกกราฟมาตรฐานของ ultralytics ของการทดลองที่ดีที่สุด
    copies = {os.path.join(TRAIN_DIR, best["run"], "results.png"): "results.png",
              os.path.join(VAL_DIR, best["run"], "confusion_matrix.png"): "confusion_matrix.png",
              os.path.join(VAL_DIR, best["run"], "BoxPR_curve.png"): "PR_curve.png"}
    for src, dst in copies.items():
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(EVAL_DIR, dst))
    print("  คัดลอก results.png, confusion_matrix.png, PR_curve.png")


if __name__ == "__main__":
    main()
