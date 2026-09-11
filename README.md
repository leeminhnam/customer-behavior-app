# 🛍️ AI E-Commerce Insight

Hệ thống **phân tích đa phương thức hành vi khách hàng** – kết hợp đánh giá văn bản (NLP/TF-IDF) và các chỉ số định lượng để dự đoán khả năng khách hàng sẽ đề xuất sản phẩm hay không. Người dùng có thể lựa chọn giữa hai mô hình: **Logistic Regression (Machine Learning)** và **Deeper MLP (Deep Learning)**.

---

## 📸 Giao diện hệ thống

> Giao diện 2 cột: Form nhập liệu bên trái, kết quả dự đoán trực quan bên phải.

---

## 🧠 Công nghệ sử dụng

### Backend
| Công nghệ | Phiên bản | Vai trò |
|---|---|---|
| **FastAPI** | latest | REST API framework |
| **Uvicorn** | latest | ASGI server |
| **scikit-learn** | 1.3.0 | Logistic Regression, TF-IDF, StandardScaler |
| **PyTorch** | latest | Deep Learning (MLP) |
| **NumPy / Pandas** | latest | Xử lý dữ liệu số học |
| **SciPy** | latest | Sparse matrix cho multimodal fusion |
| **Joblib** | latest | Serialize/load model `.pkl` |

### Frontend
| Công nghệ | Phiên bản | Vai trò |
|---|---|---|
| **React** | ^19.2.8 | UI framework |
| **Vite** | ^8.2.2 | Build tool / Dev server |
| **CSS Modules** | — | Styling tùy chỉnh |
| **Inter (Google Fonts)** | — | Typography |

### Mô hình AI
| Mô hình | Kỹ thuật | Đặc trưng đầu vào | Accuracy |
|---|---|---|---|
| **Logistic Regression** | Multimodal: TF-IDF (3.000 chiều) + Tabular | `review_text`, `rating`, `helpfulness`, `total_feedback_count` | ~96.3% |
| **Deeper MLP** | Neural Network 3 tầng (d→64→32→2) | 15 đặc trưng bảng (user + product) | ~90%+ |

---

## 📂 Cấu trúc thư mục

```
customer-behavior-app/
├── backend/
│   ├── Logistic_Regression/                     # Mô hình Machine Learning
│   │   ├── customer_behavior_prediction.ipynb   # Notebook huấn luyện LR
│   │   ├── best_ecom_model.pkl                  # Model đã lưu (joblib)
│   │   ├── tfidf_ecom.pkl                       # TF-IDF vectorizer đã fit
│   │   ├── scaler_ecom.pkl                      # StandardScaler đã fit
│   │   └── ecom_model_metadata.json             # Metadata (tên cột, vocab size, ...)
│   ├── dl/                                      # Mô hình Deep Learning
│   │   ├── customer_preference_prediction_mlp.ipynb  # Notebook huấn luyện MLP
│   │   └── sephora_preference_mlp_best.pth      # Model weights (PyTorch)
│   ├── DATA/                                    # Dataset
│   │   ├── product_info.csv                     # Dataset sản phẩm (từ Kaggle) ⬇️
│   │   └── reviews.csv                          # Dataset đánh giá (từ Kaggle) ⬇️ (gitignored – quá lớn)
│   ├── main.py                                  # FastAPI entry point
│   └── requirements.txt                         # Thư viện Python
├── frontend/
│   ├── src/
│   │   ├── App.jsx                              # Giao diện chính React
│   │   └── App.css                              # Stylesheet
│   ├── index.html                               # HTML entry point
│   └── package.json
└── README.md
```

---

## 📥 Tải Dataset từ Kaggle

Dataset được lấy từ: **[Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)**

Sau khi tải về, giải nén và đặt **2 file** sau vào thư mục `backend/DATA/`:

```
backend/
└── DATA/
    ├── product_info.csv     ← đặt ở đây
    └── reviews.csv          ← đặt ở đây  (file ~270MB, đã gitignore)
```

> ⚠️ **Lưu ý:** File `reviews.csv` có dung lượng khoảng ~270MB nên đã được thêm vào `.gitignore`. Bạn **bắt buộc phải tự tải** về từ Kaggle và đặt vào đúng thư mục mới có thể chạy lại notebook huấn luyện.

---

## 🚀 Cài đặt và chạy hệ thống

### 1. Clone repository

```bash
git clone https://github.com/leeminhnam/customer-behavior-app.git
cd customer-behavior-app
```

### 2. Tải Dataset

- Truy cập: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews
- Tải về và giải nén, đặt `product_info.csv` và `reviews.csv` vào thư mục `backend/DATA/`

### 3. Cài đặt Backend (Python)

```bash
cd backend

# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 4. Chạy Backend (FastAPI)

```bash
# Trong thư mục backend/
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

API sẽ khởi động tại: **http://127.0.0.1:8000**

> Kiểm tra API docs tại: http://127.0.0.1:8000/docs

### 5. Cài đặt và chạy Frontend (React + Vite)

Mở terminal mới, từ thư mục gốc:

```bash
cd frontend

# Cài đặt dependencies
npm install

# Chạy dev server
npm run dev
```

Giao diện sẽ mở tại: **http://localhost:5173**

---

## 🔌 API Endpoint

### `POST /api/predict`

**Request Body:**
```json
{
  "model_type": "logistic_regression",
  "review_text": "This product is amazing! My skin feels so hydrated.",
  "numerical_features": [5, 42.0, 1250, 18]
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `model_type` | `string` | `"logistic_regression"` hoặc `"dl"` |
| `review_text` | `string` | Nội dung đánh giá của khách hàng |
| `numerical_features` | `float[]` | `[rating, price_usd, loves_count, helpful_votes]` |

**Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "label": "ĐỀ XUẤT SẢN PHẨM (Tích cực)",
  "confidence": 97.42,
  "recommendation_probability": 97.42,
  "model_used": "Logistic Regression",
  "processed_text": "this product is amazing my skin feels so hydrated"
}
```

---

## 📊 Kết quả đánh giá mô hình

| Chỉ số | Logistic Regression | Deeper MLP |
|---|---|---|
| **Accuracy** | 96.31% | ~90% |
| **F1-Score** | 97.55% | ~89% |
| **ROC-AUC** | 99.15% | ~94% |

---

## 👨‍💻 Tác giả

Bài tập môn **Phát triển các hệ thống thông minh** – PTIT Học kỳ 7
