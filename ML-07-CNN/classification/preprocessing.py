"""
preprocessing.py
ขั้นที่ 2 : ย่อรูปให้เท่ากัน แปลงเป็น RGB เก็บเป็น uint8

เรื่องลำดับสี BGR กับ RGB
    OpenCV อ่านรูปเป็น BGR แต่โค้ดนี้ใช้ Pillow ซึ่งอ่านเป็น RGB อยู่แล้ว จึงไม่ต้องสลับช่องสี
    .convert("RGB") บังคับให้ทุกรูปมี 3 ช่องสี (รูปขาวดำ / PNG โปร่งใส ก็ถูกแปลงด้วย)

ทำไมเก็บเป็น uint8 (0-255) ไม่หาร 255 ตรงนี้
    uint8 ใช้ 1 ไบต์ต่อค่า แต่ float32 ใช้ 4 ไบต์ เครื่องมี RAM 8 GB จึงเก็บแบบ uint8 ไว้ก่อน
    การ standardize ไปทำใน layer แรกของโมเดล (Normalization) ทีละ batch แทน
"""

import numpy as np
from PIL import Image

from config import DATASET_FILE, IMG_SIZE

import data_loader


def image_to_array(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.uint8)


def run():
    paths, labels, classes = data_loader.run()

    X = np.stack([image_to_array(p) for p in paths])
    y = np.array(labels, dtype=np.int64)
    np.savez_compressed(DATASET_FILE, X=X, y=y)

    print(f"\nรูปทั้งหมด {X.shape[0]} ใบ  ขนาดต่อใบ {X.shape[1]}x{X.shape[2]}x{X.shape[3]}  ชนิด {X.dtype}")
    print(f"ใช้ RAM ประมาณ {X.nbytes / 1e6:.0f} MB")
    return X, y, classes


if __name__ == "__main__":
    run()
