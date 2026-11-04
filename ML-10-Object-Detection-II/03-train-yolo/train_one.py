"""
03-train-yolo/train_one.py
เทรน YOLO 1 การทดลอง (ถูกเรียกจาก main.py ใน process แยก)

    python train_one.py --run e30
    python train_one.py --run e30 --batch 8     (ลด batch เมื่อ GPU หน่วยความจำไม่พอ)

เริ่มจากน้ำหนัก COCO (yolo11n.pt) แล้วเทรนต่อกับข้อมูลหน้ากาก = transfer learning
ถ้าเคยเทรนค้างไว้ (มี last.pt แต่ยังไม่มี result.json) จะเทรนต่อจากจุดเดิม
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultralytics import YOLO

from common import RUNS, DATA_YAML, PRETRAINED, TRAIN_DIR, RANDOM_STATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    cfg = dict(RUNS[args.run])
    if args.batch:
        cfg["batch"] = args.batch
    run_dir = os.path.join(TRAIN_DIR, args.run)
    last = os.path.join(run_dir, "weights", "last.pt")
    print(f"[Run] {args.run}  epochs={cfg['epochs']}  imgsz={cfg['imgsz']}  batch={cfg['batch']}", flush=True)

    t0 = time.time()
    if os.path.exists(last) and not args.batch:
        print("  พบ last.pt เทรนต่อจากเดิม", flush=True)
        model = YOLO(last)
        model.train(resume=True)
    else:
        model = YOLO(PRETRAINED)
        model.train(
            data=DATA_YAML,
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=0,              # GPU RTX 3050
            # 0 = โหลดข้อมูลใน process หลัก  บน Windows แต่ละ worker โหลด torch ใหม่ทั้งชุด
            # (~1 GB ต่อตัว) RAM 8 GB ไม่พอ ทำให้เครื่องใช้ swap จนบาง epoch ช้าเป็น 10 เท่า
            workers=0,
            cache="disk",          # ถอดรหัสรูป PNG ครั้งเดียวเก็บเป็น .npy เทรนเร็วขึ้นมาก และไม่กิน RAM
            seed=RANDOM_STATE,
            deterministic=True,
            patience=0,            # 0 = ไม่หยุดก่อน เทรนครบทุก epoch จริง ๆ
            project=TRAIN_DIR,
            name=args.run,
            exist_ok=True,
            plots=True,
            verbose=True,
        )
    train_time = time.time() - t0

    result = {"run": args.run, **cfg, "train_time_s": round(train_time, 1)}
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  เทรนเสร็จใน {train_time / 60:.1f} นาที", flush=True)


if __name__ == "__main__":   # จำเป็นบน Windows เพราะ dataloader ใช้ multiprocessing
    main()
