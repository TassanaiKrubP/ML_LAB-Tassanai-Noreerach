"""
main.py
รันไฟล์นี้ไฟล์เดียว จะทำงานครบทุกขั้นตามลำดับ

    python main.py

ลำดับการทำงาน
    1. data_loader     สำรวจไฟล์รูป ข้ามไฟล์เสีย
    2. preprocessing   ย่อรูป แปลงเป็น RGB ปรับสเกล 0-1
    3. split_data      แบ่ง train / validation / test
    4. nn_model        เทรนและเปรียบเทียบหลายโครงสร้าง หลายจำนวน epoch
    5. evaluate        วัดผล วาด confusion matrix และกราฟการเทรน
    6. test_nn         ทายรูปสุ่ม 4 ใบ
"""

import evaluate
import nn_model
import preprocessing
import split_data
import test_nn


def header(step, title):
    print(f"\n{'=' * 60}")
    print(f"ขั้นที่ {step} : {title}")
    print("=" * 60)


def main():
    header(1, "โหลดรูปและเตรียมข้อมูล")
    preprocessing.run()          # เรียก data_loader ให้เองข้างใน

    header(2, "แบ่ง train / validation / test")
    split_data.run()

    header(3, "เทรน Neural Network และเปรียบเทียบโครงสร้าง")
    nn_model.run()

    header(4, "วัดผลและสร้างกราฟ")
    evaluate.run()

    header(5, "ทดสอบด้วยรูปสุ่ม")
    test_nn.run()

    print("\nเสร็จสมบูรณ์  ผลลัพธ์ทั้งหมดอยู่ในโฟลเดอร์ outputs/")


if __name__ == "__main__":
    main()
