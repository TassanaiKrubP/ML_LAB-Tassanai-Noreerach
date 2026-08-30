# ML-LAB-02 : Data Preprocessing

ใบงานที่ 2 วิชา Machine Learning (04-624-201)
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มทร.ธัญบุรี

ชื่อ .................................................. รหัสนักศึกษา .......................... กลุ่ม ..........

## ชุดข้อมูลที่ใช้

`f1_drivers.csv` — ข้อมูลนักแข่ง Formula 1 จำนวน 93 คน 10 คอลัมน์
(ชื่อ, ทีม, อายุ, ส่วนสูง, จำนวนชนะ, โพเดียม, คะแนนสะสม, โพลโพซิชัน, แชมป์โลก, fastest lap)

ข้อมูลต้นฉบับสะอาดเกินไปจนไม่มีอะไรให้ทำความสะอาด จึงใช้ `make_raw_data.py`
จงใจใส่ปัญหาลงไปก่อน (ใช้ seed คงที่ ผลลัพธ์เหมือนเดิมทุกครั้ง) ได้เป็น `f1_drivers_raw.csv`
ซึ่งเป็นไฟล์ที่ใช้ทำใบงานจริง ปัญหาที่ใส่ไว้มี

- ค่าที่หายไปใน `age`, `height_cm`, `team`
- แถวซ้ำ 5 แถว
- ค่าที่เป็นไปไม่ได้ เช่น อายุ -38, อายุ 999, ส่วนสูง 17.8 ซม., จำนวนชนะ -2
- ชื่อทีมพิมพ์ไม่สม่ำเสมอ เช่น `" Ferrari "`, `"red bull"`, `"MERCEDES"`
- `career_points` เก็บเป็นข้อความเพราะมีลูกน้ำคั่นหลักพัน เช่น `"4,862"`

## โครงสร้างโฟลเดอร์

```
ML-LAB-02/
├── f1_drivers.csv              ข้อมูลต้นฉบับ
├── f1_drivers_raw.csv          ข้อมูลดิบที่ใส่ปัญหาแล้ว (สร้างอัตโนมัติ)
├── f1_drivers_clean.csv        ข้อมูลที่ทำความสะอาดแล้ว (Part 3 สร้างให้)
├── make_raw_data.py            สคริปต์สร้างข้อมูลดิบ
├── main.py                     รันทุก Part ตามลำดับ
├── 01-dataset-exploration/
│   ├── load_data.py
│   ├── main.py
│   ├── plot_result.py
│   └── outputs/
├── 02-data-visualization/
├── 03-data-cleaning/
└── 04-feature-engineering/
```

แต่ละโฟลเดอร์แยกหน้าที่เหมือนกันหมด คือ `load_data.py` อ่านข้อมูล
`plot_result.py` วาดกราฟและเซฟลง `outputs/` ส่วน `main.py` เป็นลำดับขั้นตอนของ Part นั้น

## วิธีรัน

รันทั้งหมดในครั้งเดียว

```bash
cd ML-LAB-02
python main.py
```

หรือรันแยกทีละ Part (ต้องเข้าไปในโฟลเดอร์ก่อน เพราะ path อ้างอิงจากตำแหน่งไฟล์)

```bash
cd 01-dataset-exploration
python main.py
```

Part 4 ต้องรัน Part 3 ก่อน เพราะใช้ `f1_drivers_clean.csv` ที่ Part 3 สร้างไว้

## เนื้อหาแต่ละ Part

**Part 1 — Dataset Exploration**
โหลดข้อมูล ดู shape, data types, summary statistics, missing values, duplicate records
และ class distribution ของคอลัมน์ `team`

**Part 2 — Data Visualization**
Histogram ของทุกคอลัมน์ตัวเลข, correlation heatmap และ boxplot เพื่อมองหาค่าผิดปกติ

**Part 3 — Data Cleaning**
แปลงชนิดข้อมูล ลบแถวซ้ำ แก้ค่าที่ผิด เติมค่าที่หายไป และเปรียบเทียบการเติมด้วย mean กับ median

**Part 4 — Feature Engineering**
Label Encoding, One-Hot Encoding และ Feature Scaling (StandardScaler กับ MinMaxScaler)

## ไลบรารีที่ใช้

```bash
pip install pandas numpy scikit-learn matplotlib
```
