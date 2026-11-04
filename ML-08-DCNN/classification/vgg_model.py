"""
vgg_model.py
ขั้นที่ 4 : สร้าง DCNN แบบ VGG

VGG block = Conv2D 3x3 ซ้อนกัน n ชั้น (+ BatchNorm) แล้วปิดด้วย MaxPool ลดขนาดลงครึ่งหนึ่ง
    ใช้ kernel เล็ก 3x3 ซ้อนหลายชั้นแทน kernel ใหญ่ชั้นเดียว
    -> มองเห็นพื้นที่กว้างเท่ากัน แต่พารามิเตอร์น้อยกว่า และมี ReLU คั่นหลายครั้ง (เรียนรูปแบบซับซ้อนได้มากกว่า)

ตัวเลขในลิสต์ = (จำนวน filter, จำนวนชั้น conv ใน block)
"""

import tensorflow as tf
import keras
from keras import layers

from config import IMG_SIZE, SCRATCH_IMG_SIZE, BATCH_SIZE, RANDOM_STATE

VGG16 = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]    # 13 conv  (VGG16 ของจริง)
VGG_SMALL = [(32, 2), (64, 2), (128, 3), (256, 3)]          # 10 conv  เบากว่า เหมาะกับ CPU
ARCHITECTURES = {"vgg16": VGG16, "vgg_small": VGG_SMALL}


def make_dataset(X, y, shuffle=False, batch_size=BATCH_SIZE, vgg_preprocess=False):
    """ป้อนข้อมูลทีละ batch แปลง uint8 -> float32 ทีละ batch เพื่อประหยัด RAM

    vgg_preprocess=True ใช้กับโมเดล transfer learning
        VGG16 ที่ฝึกบน ImageNet ถูกสอนด้วยรูปที่ (1) สลับสีเป็น BGR (2) ลบค่าเฉลี่ยสีของ ImageNet
        ถ้าป้อนรูปต่างจากที่มันเคยเห็น น้ำหนักที่เรียนมาจะใช้ไม่ได้เต็มที่ จึงต้องใช้ preprocess_input
        แทน Normalization ที่ fit จากข้อมูลเรา
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=RANDOM_STATE, reshuffle_each_iteration=True)
    if vgg_preprocess:
        fn = lambda a, b: (keras.applications.vgg16.preprocess_input(tf.cast(a, tf.float32)), b)
    else:
        fn = lambda a, b: (tf.cast(a, tf.float32), b)
    return ds.batch(batch_size).map(fn).prefetch(tf.data.AUTOTUNE)


def make_normalization(X_train):
    """standardize : เรียน mean / variance ของแต่ละช่องสีจาก train เท่านั้น"""
    norm = layers.Normalization(axis=-1, name="standardize")
    sample = tf.data.Dataset.from_tensor_slices(X_train).batch(64).map(
        lambda a: tf.cast(a, tf.float32))
    norm.adapt(sample)
    return norm


def make_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=RANDOM_STATE),
        layers.RandomRotation(0.1, seed=RANDOM_STATE),
        layers.RandomZoom(0.1, seed=RANDOM_STATE),
    ], name="augmentation")


def vgg_block(x, filters, n_conv, block_id):
    for i in range(n_conv):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                          name=f"block{block_id}_conv{i + 1}")(x)
        x = layers.BatchNormalization(name=f"block{block_id}_bn{i + 1}")(x)
    return layers.MaxPooling2D(2, name=f"block{block_id}_pool")(x)


def add_head(x, dense_units, n_classes):
    """หัวจำแนก : GAP -> Dense -> Dropout 0.5 -> Dense -> Dropout 0.5 -> softmax"""
    x = layers.GlobalAveragePooling2D()(x)
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
    return layers.Dense(n_classes, activation="softmax")(x)


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_scratch(arch, dense_units, norm_layer, n_classes, lr):
    """VGG เทรนเองตั้งแต่ศูนย์ (น้ำหนักเริ่มแบบสุ่ม)"""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Resizing(SCRATCH_IMG_SIZE, SCRATCH_IMG_SIZE, name="resize_64")(inputs)
    x = norm_layer(x)
    x = make_augmentation()(x)
    for block_id, (filters, n_conv) in enumerate(ARCHITECTURES[arch], start=1):
        x = vgg_block(x, filters, n_conv, block_id)
    outputs = add_head(x, dense_units, n_classes)
    return compile_model(keras.Model(inputs, outputs, name=f"{arch}_scratch"), lr)


def build_transfer(dense_units, n_classes, lr):
    """Transfer learning : ยืมชั้น conv ของ VGG16 ที่ฝึกบน ImageNet (1.2 ล้านรูป) มาใช้
    แช่แข็ง (frozen) ไม่ให้น้ำหนักเปลี่ยน แล้วเทรนเฉพาะหัวจำแนกใหม่"""
    base = keras.applications.VGG16(weights="imagenet", include_top=False,
                                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = make_augmentation()(inputs)
    x = base(x, training=False)
    outputs = add_head(x, dense_units, n_classes)
    return compile_model(keras.Model(inputs, outputs, name="vgg16_transfer"), lr)


def unfreeze_last_block(model, lr):
    """Fine-tune : ปลดล็อกเฉพาะ block สุดท้าย (block5) ของ VGG16 ให้ปรับตามข้อมูลเรา ด้วย learning rate ต่ำมาก
    ชั้นต้น ๆ (ขอบ สี พื้นผิว) ใช้ได้กับทุกรูปอยู่แล้วจึงยังแช่แข็งไว้"""
    base = model.get_layer("vgg16")
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith("block5")
    return compile_model(model, lr)


def count_trainable(model):
    return int(sum(keras.ops.size(w) for w in model.trainable_weights))


class TestAccuracyAtEpochs(keras.callbacks.Callback):
    """วัด accuracy บนชุด test ที่ epoch ที่กำหนด (เหมือน Lab 7)"""

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
