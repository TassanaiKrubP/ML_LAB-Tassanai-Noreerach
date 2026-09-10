# ML-05-SVM : Support Vector Machine

ใบงานที่ 5 วิชา Machine Learning (04-624-201)
มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

## วัตถุประสงค์

จำแนกรูปภาพด้วย Support Vector Machine และเปรียบเทียบประสิทธิภาพของ kernel 3 แบบ
คือ Linear, Polynomial และ RBF

## โครงสร้างโปรเจกต์

```
ML-05-SVM/
├── PetImages/              ชุดข้อมูลรูปภาพ (ไม่ได้อัปขึ้น Git ดู link-data.txt)
├── classification/
│   ├── config.py           ค่าตั้งต้นทั้งหมด แก้ที่ไฟล์นี้ที่เดียว
│   ├── data_loader.py      อ่านรูป แปลงเป็นตัวเลข
│   ├── split_data.py       แบ่งชุดสอน / ชุดสอบ
│   ├── preprocessing.py    StandardScaler + PCA
│   ├── svm_model.py        เทรน SVM 3 kernel
│   ├── evaluate.py         วัดผลและวาด confusion matrix
│   ├── test_svm.py         ทายรูปใบใหม่
│   ├── main.py             รันครบทุกขั้นตอน
│   └── outputs/            ผลลัพธ์ที่โปรแกรมสร้างขึ้น
├── requirements.txt
└── link-data.txt           ลิงก์ดาวน์โหลดชุดข้อมูล
```

## ขั้นตอนการใช้งาน

ติดตั้งไลบรารีที่จำเป็น

```powershell
pip install -r requirements.txt
```

ดาวน์โหลดชุดข้อมูลตาม `link-data.txt` แล้ววางไว้ตามโครงสร้างด้านบน จากนั้น

```powershell
cd classification
python main.py
```

ทายรูปใบใหม่

```powershell
python test_svm.py ..\PetImages\Cat\5.jpg
```

## ขั้นตอนการทำงาน

| ขั้น | ไฟล์ | ทำอะไร |
|---|---|---|
| 1 | data_loader.py | ย่อรูปเป็น 64x64 ขาวดำ แล้วคลี่เป็นแถวตัวเลข 4096 ตัว |
| 2 | split_data.py | แบ่งข้อมูล 80% สอน 20% สอบ แบบรักษาสัดส่วนคลาส |
| 3 | preprocessing.py | ปรับสเกลด้วย StandardScaler แล้วบีบมิติเหลือ 100 ด้วย PCA |
| 4 | svm_model.py | เทรน Linear, Polynomial, RBF แล้วเทียบ accuracy |
| 5 | evaluate.py | สร้างรายงานผลและ confusion matrix |

## ผลการทดลอง

| Kernel | Accuracy | เวลาเทรน (วินาที) |
|---|---|---|
| Linear | | |
| Polynomial | | |
| RBF | | |

(กรอกตัวเลขจริงหลังรัน `main.py` เสร็จ)

## สรุปผล

(เขียนหลังได้ผลลัพธ์จริง)
