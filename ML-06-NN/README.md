# ML-06-NN : Neural Network

ใบงานที่ 6 วิชา Machine Learning (04-624-201)
มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

## วัตถุประสงค์

จำแนกรูปภาพด้วย Neural Network และเปรียบเทียบผลของ

- จำนวน epoch ที่ใช้เทรน
- โครงสร้างเครือข่ายที่ต่างกัน (จำนวนชั้นซ่อนและจำนวนนิวรอน)

## ชุดข้อมูล

Bird / Drone / Airplane จำนวน 3 คลาส (ชุดเดียวกับใบงานที่ 5)
ไฟล์รูปไม่ได้อัปขึ้น GitHub เนื่องจากขนาดใหญ่

## สภาพแวดล้อม

ต้องใช้ **Python 3.12** เนื่องจาก TensorFlow ยังไม่รองรับ Python 3.14

```powershell
conda activate lab6
pip install -r requirements.txt
```

## ขั้นตอนการใช้งาน

```powershell
cd classification
python main.py
```

ทดสอบด้วยรูปสุ่ม 4 ใบ

```powershell
python test_nn.py
```

## โครงสร้างโปรเจกต์

```
ML-06-NN/
├── BirdVsDroneVsAirplane/     ชุดข้อมูล (ไม่ได้อัปขึ้น Git)
├── classification/
│   ├── config.py              ค่าตั้งต้นทั้งหมด
│   ├── data_loader.py         สำรวจไฟล์ ข้ามไฟล์เสีย
│   ├── preprocessing.py       ย่อรูป แปลง RGB ปรับสเกล
│   ├── split_data.py          แบ่ง train / validation / test
│   ├── nn_model.py            สร้าง เทรน เปรียบเทียบโครงสร้าง
│   ├── evaluate.py            วัดผล confusion matrix กราฟการเทรน
│   ├── test_nn.py             ทายรูปสุ่ม 4 ใบ
│   └── outputs/               ผลลัพธ์ที่โปรแกรมสร้าง
├── requirements.txt
└── README.md
```

## การแบ่งข้อมูล

| ชุด | สัดส่วน | ใช้ทำอะไร |
|---|---|---|
| train | 70% | โมเดลเรียนรู้และปรับน้ำหนักจากชุดนี้ |
| validation | 15% | ตรวจสอบทุกจบ epoch เพื่อดูอาการ overfitting |
| test | 15% | สอบครั้งเดียวตอนจบ เป็นคะแนนที่รายงาน |

## โครงสร้างที่นำมาเปรียบเทียบ

| ชื่อ | ชั้นซ่อน | Dropout |
|---|---|---|
| 1-hidden-128 | 128 | ไม่มี |
| 2-hidden-256-128 | 256, 128 | 0.3 |
| 3-hidden-512-256-128 | 512, 256, 128 | 0.3 |

## ผลการทดลอง

### เปรียบเทียบตามจำนวน epoch (val_accuracy)

| โครงสร้าง | ep 5 | ep 10 | ep 20 | ep 30 | ep 40 |
|---|---|---|---|---|---|
| 1-hidden-128 | | | | | |
| 2-hidden-256-128 | | | | | |
| 3-hidden-512-256-128 | | | | | |

### ผลบนชุด test

(กรอกหลังรัน `main.py` เสร็จ)

## สรุปผล

(เขียนหลังได้ผลลัพธ์จริง)
