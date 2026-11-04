"""
05-inference/realtime_camera.py
ตรวจจับหน้ากากแบบเรียลไทม์จากกล้องเว็บแคม  กด q เพื่อออก

    python realtime_camera.py              กล้องตัวแรกของเครื่อง (source 0)
    python realtime_camera.py --source 1   กล้องตัวที่สอง

ต้องรันเองต่อหน้ากล้อง (มีหน้าต่างภาพเด้งขึ้นมา)
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2

from common import CONF_THRESHOLD
from detect_images import load_best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0)
    args = parser.parse_args()

    model, imgsz = load_best_model()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("เปิดกล้องไม่ได้ ลองเปลี่ยน --source เป็น 1 หรือปิดโปรแกรมอื่นที่ใช้กล้องอยู่")
        return

    print("กด q ที่หน้าต่างภาพเพื่อออก")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=CONF_THRESHOLD, device=0, verbose=False)[0]
        annotated = r.plot()

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(annotated, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Face mask detection (press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
