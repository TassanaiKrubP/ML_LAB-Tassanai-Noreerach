"""
cnn_model.py
ขั้นที่ 4 : สร้าง CNN แบบปรับโครงสร้างได้ (จำนวนชั้น conv, จำนวน filter, นิวรอน, ขนาด kernel)

โครงสร้าง (ตามแบบอาจารย์)
    Normalization            standardize ค่าพิกเซล (mean 0, std 1 ต่อช่องสี) fit จาก train เท่านั้น
    Augmentation             พลิก/หมุน/ซูมรูปแบบสุ่มตอนเทรน ให้โมเดลเห็นรูปหลากหลายขึ้น
    [Conv2D + BatchNorm + MaxPool] x n_blocks
        Conv2D   เลื่อน filter ขนาด kernel x kernel ไปทั่วรูป หาลวดลาย (ขอบ มุม พื้นผิว)
        ReLU     ตัดค่าติดลบเป็น 0 ให้โมเดลเรียนรู้ความสัมพันธ์ที่ไม่ใช่เส้นตรงได้
        MaxPool  ย่อ feature map ลงครึ่งหนึ่ง เก็บค่าที่เด่นที่สุด ลดการคำนวณ
    GlobalAveragePooling2D   เฉลี่ย feature map แต่ละแผ่นเหลือ 1 ค่า (แทน Flatten ใช้พารามิเตอร์น้อยกว่า)
    Dropout -> Dense -> Dropout -> softmax (3 คลาส)
"""

import tensorflow as tf
import keras
from keras import layers

from config import IMG_SIZE, BATCH_SIZE, LEARNING_RATE, RANDOM_STATE


def make_dataset(X, y, shuffle=False, batch_size=BATCH_SIZE):
    """ป้อนข้อมูลทีละ batch และแปลง uint8 -> float32 ทีละ batch เพื่อประหยัด RAM"""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=RANDOM_STATE, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(lambda a, b: (tf.cast(a, tf.float32), b))
    return ds.prefetch(tf.data.AUTOTUNE)


def make_normalization(X_train):
    """เรียนค่า mean / variance ของแต่ละช่องสีจาก train เท่านั้น (ไม่แอบดู val/test)"""
    norm = layers.Normalization(axis=-1, name="standardize")
    sample = tf.data.Dataset.from_tensor_slices(X_train).batch(64).map(
        lambda a: tf.cast(a, tf.float32))
    norm.adapt(sample)
    return norm


def build_model(config, norm_layer, n_classes):
    """
    config ตัวอย่าง
        {"filters": [32, 64, 128], "dense_units": 128, "kernel_size": 3}
    จำนวนชั้น conv = len(filters)
    """
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", seed=RANDOM_STATE),
        layers.RandomRotation(0.1, seed=RANDOM_STATE),
        layers.RandomZoom(0.1, seed=RANDOM_STATE),
    ], name="augmentation")

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = norm_layer(inputs)
    x = augmentation(x)

    k = config.get("kernel_size", 3)
    for i, f in enumerate(config["filters"]):
        x = layers.Conv2D(f, k, padding="same", activation="relu", name=f"conv{i + 1}")(x)
        x = layers.BatchNormalization(name=f"bn{i + 1}")(x)
        x = layers.MaxPooling2D(name=f"pool{i + 1}")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(config["dense_units"], activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn")
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


class TestAccuracyAtEpochs(keras.callbacks.Callback):
    """วัด accuracy บนชุด test ที่ epoch ที่กำหนด เพื่อทำกราฟ accuracy vs epochs ในการเทรนรอบเดียว"""

    def __init__(self, test_ds, checkpoints):
        super().__init__()
        self.test_ds = test_ds
        self.checkpoints = set(checkpoints)
        self.results = {}

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep in self.checkpoints:
            loss, acc = self.model.evaluate(self.test_ds, verbose=0)
            self.results[ep] = {"test_accuracy": float(acc), "test_loss": float(loss),
                                "val_accuracy": float(logs.get("val_accuracy", 0))}
            print(f"  [checkpoint] epoch {ep}: test accuracy = {acc:.4f}", flush=True)
