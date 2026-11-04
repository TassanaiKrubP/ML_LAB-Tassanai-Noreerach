"""
05-inference/detect_video.py
ตรวจจับหน้ากากในวิดีโอ ทีละเฟรม แล้วบันทึกวิดีโอใหม่ที่มีกรอบ

    python detect_video.py                       ไม่มีวิดีโอ -> สร้างวิดีโอสั้นจากรูป test ให้ก่อน
    python detect_video.py --source clip.mp4     ใช้วิดีโอของเราเอง

หมายเหตุ  cv2.VideoWriter / VideoCapture บน Windows เปิด path ภาษาไทยไม่ได้
         จึงเขียน/อ่านวิดีโอในโฟลเดอร์ชั่วคราว (TEMP) ก่อน แล้วค่อยย้ายมาไว้ที่ outputs/
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from common import YOLO_DATA_DIR, CONF_THRESHOLD, imread, imwrite
from detect_images import load_best_model

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
TMP = tempfile.gettempdir()
FRAME_W, FRAME_H, FPS = 640, 480, 10


def letterbox(img, w=FRAME_W, h=FRAME_H):
    """ย่อรูปให้พอดีเฟรมโดยไม่บิดสัดส่วน เติมขอบสีเทา"""
    scale = min(w / img.shape[1], h / img.shape[0])
    resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    y0, x0 = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def build_demo_video(path, n_images=30, seconds_each=1.5):
    """สร้างวิดีโอจากรูป test (รูปละ 1.5 วินาที ค่อย ๆ ซูมเข้าเล็กน้อยให้เหมือนกล้องเคลื่อน)"""
    img_dir = os.path.join(YOLO_DATA_DIR, "images", "test")
    files = sorted(os.listdir(img_dir))[:n_images]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (FRAME_W, FRAME_H))
    n_frames = int(FPS * seconds_each)
    for name in files:
        img = imread(os.path.join(img_dir, name))
        for k in range(n_frames):
            zoom = 1.0 + 0.08 * k / n_frames
            h, w = img.shape[:2]
            ch, cw = int(h / zoom), int(w / zoom)
            crop = img[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
            writer.write(letterbox(crop))
    writer.release()
    print(f"  สร้างวิดีโอทดสอบจากรูป test {len(files)} รูป ({len(files) * seconds_each:.0f} วินาที)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[Step 5b] Detecting masks in a video...")
    src_tmp = os.path.join(TMP, "lab10_input.mp4")
    if args.source:
        shutil.copy2(args.source, src_tmp)
    else:
        build_demo_video(src_tmp)
        shutil.copy2(src_tmp, os.path.join(OUT_DIR, "test_video.mp4"))

    model, imgsz = load_best_model()
    cap = cv2.VideoCapture(src_tmp)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_tmp = os.path.join(TMP, "lab10_output.mp4")
    writer = cv2.VideoWriter(out_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=CONF_THRESHOLD, device=0, verbose=False)[0]
        annotated = r.plot()
        writer.write(annotated)
        if n % 60 == 7:          # เก็บบางเฟรมเป็นรูปไว้ใส่ README
            imwrite(os.path.join(OUT_DIR, f"video_frame_{n:04d}.jpg"), annotated)
        n += 1
    cap.release()
    writer.release()
    elapsed = time.time() - t0

    shutil.move(out_tmp, os.path.join(OUT_DIR, "annotated_video.mp4"))
    print(f"  ประมวลผล {n} เฟรม ใน {elapsed:.1f} วินาที  ({n / elapsed:.1f} FPS)")
    print("  บันทึก outputs/annotated_video.mp4")
    with open(os.path.join(OUT_DIR, "video_speed.txt"), "w", encoding="utf-8") as f:
        f.write(f"frames={n}\nseconds={elapsed:.1f}\nfps={n / elapsed:.1f}\n")


if __name__ == "__main__":
    main()
