from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import os

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
metrics_path = os.path.join(os.path.dirname(__file__), "metrics.json")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load metrics
with open(metrics_path, "r") as f:
    model_metrics = json.load(f)

app = FastAPI(
    title="Boston Housing Regression API - Lab 5",
    description="MLOps API for predicting house prices using the Boston dataset",
    version="1.0"
)

class HouseFeatures(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {
        "message": "Boston Housing Regression API (Lab 5) is running!",
        "metrics": model_metrics,
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    return model_metrics

@app.post("/predict")
def predict(data: HouseFeatures):
    if len(data.features) != 13:
        return {"error": f"Model expects exactly 13 features, but got {len(data.features)}"}
    
    # Scale and predict
    scaled_features = scaler.transform([data.features])
    prediction = model.predict(scaled_features)[0]
    
    return {
        "predicted_price": round(float(prediction), 4)
    }
