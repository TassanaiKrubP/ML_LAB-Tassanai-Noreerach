"""
voc_to_yolo.py
แปลงกรอบจากรูปแบบ PASCAL VOC (ไฟล์ XML) เป็นรูปแบบ YOLO (ไฟล์ txt)

    VOC  : x_min, y_min, x_max, y_max      หน่วยเป็นพิกเซล (มุมซ้ายบน / มุมขวาล่าง)
    YOLO : class x_center y_center width height   หารด้วยขนาดรูป ให้อยู่ในช่วง 0-1

ตัวอย่าง รูปกว้าง 512 สูง 366  กรอบ (79, 105, 109, 142)
    x_center = (79 + 109) / 2 / 512 = 0.1836
    y_center = (105 + 142) / 2 / 366 = 0.3374
    width    = (109 - 79) / 512     = 0.0586
    height   = (142 - 105) / 366    = 0.1011

รันไฟล์นี้ตรง ๆ เพื่อทดสอบการแปลงกับตัวอย่างสังเคราะห์  python voc_to_yolo.py
"""

import xml.etree.ElementTree as ET


def parse_voc(xml_path):
    """อ่าน XML 1 ไฟล์ -> ชื่อรูป, กว้าง, สูง, รายการ (ชื่อคลาส, xmin, ymin, xmax, ymax)"""
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    w, h = int(size.findtext("width")), int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        coords = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        objects.append((obj.findtext("name"), *coords))
    return filename, w, h, objects


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """แปลงกรอบ 1 กรอบ  ตัดค่าที่เลยขอบรูปออก แล้วหารด้วยขนาดรูป"""
    xmin, xmax = max(0.0, xmin), min(float(img_w), xmax)
    ymin, ymax = max(0.0, ymin), min(float(img_h), ymax)
    xc = (xmin + xmax) / 2 / img_w
    yc = (ymin + ymax) / 2 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return xc, yc, bw, bh


def to_yolo_lines(objects, img_w, img_h, classes):
    """คืนรายการบรรทัดของไฟล์ label  ข้ามกรอบที่กว้างหรือสูงเป็น 0"""
    lines = []
    for name, xmin, ymin, xmax, ymax in objects:
        xc, yc, bw, bh = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        if bw > 0 and bh > 0:
            lines.append(f"{classes.index(name)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _self_test():
    """ทดสอบกับ XML สังเคราะห์ที่รู้คำตอบล่วงหน้า"""
    import os
    import tempfile

    xml = """<annotation><filename>demo.png</filename>
      <size><width>200</width><height>100</height><depth>3</depth></size>
      <object><name>with_mask</name><bndbox><xmin>50</xmin><ymin>25</ymin><xmax>150</xmax><ymax>75</ymax></bndbox></object>
      <object><name>without_mask</name><bndbox><xmin>-10</xmin><ymin>0</ymin><xmax>20</xmax><ymax>50</ymax></bndbox></object>
    </annotation>"""
    path = os.path.join(tempfile.mkdtemp(), "demo.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)

    filename, w, h, objects = parse_voc(path)
    lines = to_yolo_lines(objects, w, h, ["with_mask", "without_mask"])
    print(f"{filename} {w}x{h}")
    for line in lines:
        print("  ", line)
    # กรอบแรกอยู่กลางรูปพอดี กว้างครึ่งรูป สูงครึ่งรูป
    assert lines[0] == "0 0.500000 0.500000 0.500000 0.500000"
    # กรอบที่สองเลยขอบซ้าย ต้องถูกตัดที่ x = 0  -> กว้าง 20 px
    assert lines[1] == "1 0.050000 0.250000 0.100000 0.500000"
    print("ผ่าน : แปลง VOC -> YOLO ถูกต้อง")


if __name__ == "__main__":
    _self_test()
