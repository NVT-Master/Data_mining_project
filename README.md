# Crop Yield Prediction - Data Mining Project

## 📋 Mô Tả Dự Án

Dự án này xây dựng một pipeline khai phá dữ liệu hoàn chỉnh để dự báo năng suất cây trồng (Crop Yield Prediction) sử dụng dataset từ Kaggle.

**Mục tiêu chính:**
- Khám phá và phân tích dữ liệu liên quan đến năng suất cây trồng
- Xây dựng các mô hình dự báo cả baseline và advanced
- Tìm ra các yếu tố quan trọng ảnh hưởng đến năng suất
- Đưa ra khuyến nghị canh tác

## 📊 Dataset

**Nguồn:** [Kaggle - Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

**Các file CSV:**
- `yield.csv` - Năng suất cây trồng chính
- `pesticides.csv` - Dữ liệu sử dụng thuốc trừ sâu
- `rainfall.csv` - Dữ liệu lượng mưa
- `temperature.csv` - Dữ liệu nhiệt độ

## 🏗️ Cấu Trúc Dự Án

```
DATA_MINING_PROJECT/
├── README.md                 # File này
├── requirements.txt          # Dependencies
├── configs/
│   └── params.yaml          # Cấu hình tham số
├── data/
│   ├── raw/                 # Dữ liệu gốc
│   └── processed/           # Dữ liệu đã xử lý
├── notebooks/               # Các Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_clustering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── src/                     # Source code chính
│   ├── data/
│   │   ├── loader.py       # Load dữ liệu
│   │   └── cleaner.py      # Xử lý dữ liệu
│   ├── features/
│   │   └── builder.py      # Feature engineering
│   ├── mining/
│   │   ├── association.py  # Association rules
│   │   └── clustering.py   # Clustering
│   ├── models/
│   │   └── regression.py   # Các mô hình regression
│   ├── evaluation/
│   │   ├── metrics.py      # Tính toán metrics
│   │   └── report.py       # Tạo báo cáo
│   └── visualization/
│       └── plots.py        # Các hàm vẽ biểu đồ
├── scripts/
│   └── run_pipeline.py     # Chạy toàn bộ pipeline
├── outputs/
│   ├── figures/            # Lưu hình ảnh
│   ├── tables/             # Lưu bảng dữ liệu
│   └── models/             # Lưu các mô hình
└── reports/
    └── final_report.md     # Báo cáo cuối cùng
```

## 🚀 Quick Start

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị Dữ Liệu

Tải dataset từ Kaggle và đặt các file CSV vào thư mục `data/raw/`:
```
data/raw/
├── yield.csv
├── pesticides.csv
├── rainfall.csv
└── temp.csv
```

### 3. Chạy Pipeline Hoàn Chỉnh

```bash
python scripts/run_pipeline.py
```

### 4. Xem Kết Quả

- **Notebooks:** Mở các file `.ipynb` trong thư mục `notebooks/`
- **Biểu đồ:** Xem các file `.png` trong `outputs/figures/`
- **Báo cáo:** Xem `reports/final_report.md`

## 📈 Pipeline Phân Tích

```
1. DATA LOADING
   ↓
2. EXPLORATORY DATA ANALYSIS (EDA)
   ↓
3. DATA PREPROCESSING & FEATURE ENGINEERING
   ↓
4. ASSOCIATION RULE MINING
   ↓
5. CLUSTERING
   ↓
6. FEATURE SELECTION
   ↓
7. MODEL TRAINING (Baseline & Advanced)
   ↓
8. MODEL EVALUATION & COMPARISON
   ↓
9. VISUALIZATION & REPORTING
```

## 🔬 Các Phương Pháp Sử Dụng

### Preprocessing
- Xử lý missing values (mean, median, forward fill)
- Xử lý outliers (IQR, Z-score)
- Scaling (StandardScaler, MinMaxScaler)
- Encoding (OneHotEncoder, LabelEncoder)

### Feature Engineering
- Tạo feature mới từ dữ liệu hiện có
- Chọn feature quan trọng (SelectKBest)
- Feature importance từ Random Forest

### Mining
- **Association Rules:** Apriori algorithm
- **Clustering:** KMeans, DBSCAN

### Modeling
- **Baseline:** Linear Regression, Ridge Regression
- **Advanced:** Random Forest, XGBoost

### Evaluation
- Metrics: MAE, RMSE, R², MAPE
- Cross-validation
- Residual analysis

## 📊 Kết Quả Chính

*Sẽ cập nhật sau khi chạy pipeline*

### Model Performance Comparison

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|-----|------|
| Linear Regression | ... | ... | ... | ... |
| Ridge Regression | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... |

### Top Features
*Sẽ hiển thị các feature quan trọng nhất*

### Key Insights
*Sẽ cập nhật từ kết quả phân tích*

## 💡 Insights & Recommendations

*Sẽ cập nhật sau phân tích*

## 📚 Công Nghệ Sử Dụng

- **Python 3.8+**
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Mining:** MLxtend
- **Config:** PyYAML

## 👥 Tác Giả

Data Mining Project - BTL Năm 3 Kì 2

## 📝 Ghi Chú

- Tất cả code đều có comment chi tiết
- Theo chuẩn PEP 8 Python
- Sử dụng modular design cho dễ bảo trì
- Có thể mở rộng cho các dữ liệu khác

## 📞 Liên Hệ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ.

---

**Last Updated:** 2026-03-14
