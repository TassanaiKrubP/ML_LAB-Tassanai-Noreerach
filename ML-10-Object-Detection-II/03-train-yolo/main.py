"""
03-train-yolo/main.py
ขั้นที่ 3 : เทรน YOLO ทุกการทดลองในตาราง RUNS (common.py) ทีละ process

    epoch      : e10 / e30 / e60                       (imgsz 640, batch 16)
    ขนาดรูป/batch : s320_b8 / s320_b16 / s640_b8 / s640_b16(= e30)   (30 epoch)

การทดลองที่มี result.json แล้วจะถูกข้าม (รันซ้ำได้ ทำต่อจากเดิม)
ถ้า GPU หน่วยความจำไม่พอ (CUDA out of memory) จะลองใหม่ด้วย batch ครึ่งหนึ่ง
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import RUNS, TRAIN_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "outputs", "logs")


def run_one(name, batch=None):
    if os.path.exists(os.path.join(TRAIN_DIR, name, "result.json")):
        print(f"  ข้าม {name} (เทรนเสร็จแล้ว)")
        return True

    cmd = [sys.executable, "-u", os.path.join(HERE, "train_one.py"), "--run", name]
    if batch:
        cmd += ["--batch", str(batch)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env, cwd=HERE)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()

    if proc.returncode != 0:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            oom = "out of memory" in f.read().lower()
        current = batch or RUNS[name]["batch"]
        if oom and current > 2:
            print(f"  !! {name} GPU หน่วยความจำไม่พอ ลองใหม่ด้วย batch {current // 2}")
            return run_one(name, current // 2)
        print(f"  !! {name} ล้มเหลว ดู {log_path}")
        return False
    return True


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print("[Step 3] Training YOLO experiments...")
    failed = [name for name in RUNS if not run_one(name)]
    if failed:
        print(f"การทดลองที่ล้มเหลว : {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
