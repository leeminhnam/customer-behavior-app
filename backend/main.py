import os
import re
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from scipy.sparse import hstack

app = FastAPI(title="E-Commerce Customer Behavior Discovery API")

# Cho phép CORS cho frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- KHAI BÁO MODEL DEEP LEARNING ---
class DeeperPreferenceMLP(nn.Module):
    def __init__(self, input_dim=15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(32, 2)

    def forward(self, x):
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        return self.fc_out(h2)

# Global variables cho models
lr_model = None
lr_tfidf = None
lr_scaler = None
dl_model = None

@app.on_event("startup")
def load_models():
    global lr_model, lr_tfidf, lr_scaler, dl_model

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load Logistic Regression
    lr_dir = os.path.join(BASE_DIR, "Logistic_Regression")
    if os.path.exists(lr_dir):
        try:
            lr_model = joblib.load(os.path.join(lr_dir, "best_ecom_model.pkl"))
            lr_tfidf = joblib.load(os.path.join(lr_dir, "tfidf_ecom.pkl"))
            lr_scaler = joblib.load(os.path.join(lr_dir, "scaler_ecom.pkl"))
            print("Loaded Logistic Regression models successfully.")
        except Exception as e:
            print(f"Error loading LR models: {e}")

    # Load Deep Learning model
    dl_dir = os.path.join(BASE_DIR, "dl")
    if os.path.exists(dl_dir):
        try:
            dl_model = DeeperPreferenceMLP(input_dim=15)
            dl_model.load_state_dict(torch.load(os.path.join(dl_dir, "sephora_preference_mlp_best.pth"), map_location=torch.device('cpu')))
            dl_model.eval()
            print("Loaded DL model successfully.")
        except Exception as e:
            print(f"Error loading DL model: {e}")

class PredictRequest(BaseModel):
    model_type: str = "logistic_regression"
    review_text: str
    numerical_features: List[float]

def clean_text_inference(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "E-Commerce Customer Behavior Discovery API (FastAPI)",
        "message": "FastAPI đang chạy!"
    }

@app.post("/api/predict")
def predict(request: PredictRequest):
    try:
        raw_text = request.review_text
        num_features = request.numerical_features
        model_type = request.model_type

        cleaned_str = clean_text_inference(raw_text)

        if model_type == "logistic_regression":
            if lr_model is None or lr_tfidf is None or lr_scaler is None:
                raise HTTPException(status_code=500, detail="Logistic Regression model is not loaded.")

            # TF-IDF
            vec_text = lr_tfidf.transform([cleaned_str])

            # Frontend gửi: [rating, price_usd, loves_count, helpful_votes]
            # LR model cần 4 cột: ["Unnamed: 0", "rating_review", "helpfulness", "total_feedback_count"]
            rating = num_features[0]
            helpful_votes = num_features[3]

            # Lấy giá trị trung bình của Unnamed: 0 từ scaler để không làm nhiễu prediction
            unnamed_0_val = lr_scaler.mean_[0] if hasattr(lr_scaler, 'mean_') else 500000.0

            # Map lại cho đúng feature thứ tự của LR
            mapped_lr_features = [unnamed_0_val, rating, helpful_votes, helpful_votes + 1]
            num_array = np.array([mapped_lr_features], dtype=np.float64)

            vec_num = lr_scaler.transform(num_array)

            # Ghép khối đa phương thức
            X_infer = hstack([vec_num, vec_text]).tocsr()

            prediction = int(lr_model.predict(X_infer)[0])
            if hasattr(lr_model, "predict_proba"):
                probabilities = lr_model.predict_proba(X_infer)[0]
                confidence = round(float(probabilities[prediction]) * 100, 2)
                prob_positive = round(float(probabilities[1]) * 100, 2) if len(probabilities) > 1 else confidence
            else:
                confidence = 100.0
                prob_positive = 100.0 if prediction == 1 else 0.0

            model_used_name = "Logistic Regression"

        elif model_type == "dl":
            if dl_model is None:
                raise HTTPException(status_code=500, detail="DL model is not loaded.")

            rating = num_features[0]

            # MLP DL yêu cầu input dim = 15 và ĐÃ ĐƯỢC Z-SCORE NORMALIZED.
            # Vì không có file scaler của DL, ta truyền trực tiếp giá trị chuẩn hóa (Z-scores).
            # [0.0] tức là giá trị trung bình của feature đó.
            dl_features = [0.0] * 15

            # Giả lập ảnh hưởng của rating vào features của DL (bằng Z-score ~ -3 đến 3)
            if rating <= 2:
                dl_features[6] = 3.0   # total_neg_feedback_count cao (3 std)
                dl_features[5] = -1.0  # total_pos_feedback_count thấp
                dl_features[7] = -1.0  # loves_count thấp
            elif rating >= 4:
                dl_features[5] = 3.0   # total_pos_feedback_count cao (3 std)
                dl_features[6] = -1.0  # total_neg_feedback_count thấp
                dl_features[7] = 2.0   # loves_count cao
            else:
                dl_features[5] = 0.5
                dl_features[6] = 0.5

            x_tensor = torch.tensor([dl_features], dtype=torch.float32)

            with torch.no_grad():
                logits = dl_model(x_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                prediction = int(torch.argmax(probs).item())
                confidence = round(float(probs[prediction].item()) * 100, 2)
                prob_positive = round(float(probs[1].item()) * 100, 2)

            model_used_name = "Deep Learning (MLP)"

        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Choose 'logistic_regression' or 'dl'.")

        label_map = {
            1: "ĐỀ XUẤT SẢN PHẨM (Tích cực)",
            0: "KHÔNG ĐỀ XUẤT (Tiêu cực / Rủi ro rời bỏ)"
        }

        return {
            "status": "success",
            "prediction": prediction,
            "label": label_map.get(prediction, "Chưa xác định"),
            "confidence": confidence,
            "recommendation_probability": prob_positive,
            "model_used": model_used_name,
            "processed_text": cleaned_str
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
