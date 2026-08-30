# ML-04-K-Nearest Neighbors (KNN) (อัพแค่outputมา ส่วนdatasetไม่ได้เอาขึ้นมาด้วย)

Build a simple KNN pipeline using Python, including image data loading, preprocessing, feature scaling, model training, evaluation, and prediction.

# Data

Drone classification Dataset (AirplaneVsDroneVsBird): https://www.kaggle.com/datasets/maryamlsgumel/drone-detection-dataset

Three classes of flying objects photographed against the sky: **Aeroplanes**, **Birds** and **Drones**.

# Structure

```text
ML-04-KNN/
│
├── dataset/
│   ├── Aeroplanes/
│   │   ├── 0.jpg
│   │   └── ...
│   ├── Birds/
│   │   ├── 0.jpg
│   │   └── ...
│   └── Drones/
│       ├── 0.jpg
│       └── ...
│
├── classification/
│   ├── main.py
│   ├── test_knn.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── knn_model.py
│   ├── evaluate.py
│   └── outputs/
│       ├── images.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── scaler.pkl
│       ├── knn_model.pkl
│       ├── 01_k_curve.png
│       ├── 02_confusion_matrix.png
│       ├── prediction_sample.png
│       └── predictions.csv
│
├── clustering/
│   ├── main.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── kmeans_model.py
│   ├── knn_tools.py
│   ├── visualize.py
│   └── outputs/
│       ├── classes.json
│       ├── transformer.pkl
│       ├── kmeans_model.pkl
│       ├── cluster_knn.pkl
│       ├── 01_elbow.png
│       ├── 02_clusters.png
│       ├── cluster_summary.csv
│       └── clustered_images.csv
│
├── requirements.txt
└── link-data.txt
```

# How to run

```bash
pip install -r requirements.txt

cd classification
python main.py
python test_knn.py

cd ../clustering
python main.py
```

# Settings

| Setting | Value | Reason |
| --- | --- | --- |
| `IMG_SIZE` | 32 | 1024 features per image. Larger sizes trigger the curse of dimensionality, which KNN is especially sensitive to. |
| `MAX_PER_CLASS` | 500 | Keeps the three classes balanced. KNN decides by majority vote, so an over-represented class biases every vote. |
| `TEST_SIZE` | 0.2 | Standard 80/20 split, stratified to preserve class ratios. |
| `K_VALUES` | 3, 5, 7 | Required by the lab sheet. |
| `PCA_COMPONENTS` | 50 | Used in the clustering pipeline to make distances meaningful again. |

# Summary

The project applies KNN to a three-class image recognition problem. Images are loaded from class directories, converted to grayscale, resized, flattened into feature vectors and standardized before training. Models are trained with k = 3, 5 and 7, compared by test accuracy, and evaluated with a classification report and a confusion matrix. The value of k is then swept from 1 to 31 to show the trade-off between overfitting at small k and underfitting at large k.

The clustering half applies KMeans to the same features without using the labels, uses the elbow method to choose the number of clusters, and compares the discovered groups against the true classes to see whether the natural structure of the raw pixel data lines up with the classes of interest.
