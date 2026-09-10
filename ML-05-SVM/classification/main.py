"""
main.py
รันไฟล์นี้ไฟล์เดียว จะทำงานครบทั้ง 5 ขั้นตามลำดับ

    python main.py

ลำดับการทำงาน
    1. data_loader    อ่านรูป -> แปลงเป็นตัวเลข
    2. split_data     แบ่งชุดสอน / ชุดสอบ
    3. preprocessing  Standardize + PCA
    4. svm_model      เทรน 3 kernel เทียบ accuracy
    5. evaluate       วัดผล + วาด confusion matrix
"""

import data_loader
import evaluate
import preprocessing
import split_data
import svm_model


def header(step, title):
    print(f"\n{'=' * 55}")
    print(f"ขั้นที่ {step} : {title}")
    print("=" * 55)


def main():
    header(1, "โหลดและแปลงรูปเป็นตัวเลข")
    data_loader.run()

    header(2, "แบ่งชุดสอนและชุดสอบ")
    split_data.run()

    header(3, "Standardize feature และลดมิติด้วย PCA")
    preprocessing.run()

    header(4, "เทรน SVM ทั้ง 3 kernel")
    svm_model.run()

    header(5, "วัดผลและสร้าง confusion matrix")
    evaluate.run()

    print("\nเสร็จสมบูรณ์  ผลลัพธ์ทั้งหมดอยู่ในโฟลเดอร์ outputs/")


if __name__ == "__main__":
    main()
