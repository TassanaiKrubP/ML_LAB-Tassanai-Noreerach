"""
main.py
Lab 10 : Object Detection ด้วย YOLO  รันขั้น 01 -> 05 ต่อกันอัตโนมัติ (ทีละ process)

    .\\.venv-yolo\\Scripts\\python main.py

ขั้นที่เสร็จแล้วจะถูกข้าม ถ้าเครื่องดับกลางทาง รันใหม่ก็ทำต่อได้
การทดลองกล้องเว็บแคม (05-inference/realtime_camera.py) ต้องรันเอง ไม่อยู่ในนี้
"""

import os
import subprocess
import sys
import time

from common import ROOT_DIR, DATA_YAML

STEPS = [
    ("01-prepare-data/main.py",        lambda: os.path.exists(DATA_YAML)),
    ("02-pretrained-yolo/main.py",     lambda: os.path.exists(os.path.join(
        ROOT_DIR, "02-pretrained-yolo", "outputs", "pretrained_predictions.png"))),
    ("03-train-yolo/main.py",          lambda: False),   # ข้ามการทดลองที่เสร็จแล้วเอง
    ("04-evaluate/main.py",            lambda: False),
    ("04-evaluate/plot_result.py",     lambda: False),
    ("05-inference/detect_images.py",  lambda: False),
    ("05-inference/detect_video.py",   lambda: False),
]


def main():
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for script, done in STEPS:
        print(f"\n===== {script} =====", flush=True)
        if done():
            print("  เสร็จแล้ว ข้าม")
            continue
        path = os.path.join(ROOT_DIR, script)
        ret = subprocess.run([sys.executable, "-u", path], cwd=os.path.dirname(path), env=env).returncode
        if ret != 0:
            print(f"!! {script} ล้มเหลว (exit {ret}) หยุดที่ขั้นนี้")
            sys.exit(ret)
    print(f"\nเสร็จทั้งหมดใน {(time.time() - t0) / 60:.1f} นาที")


if __name__ == "__main__":
    main()
