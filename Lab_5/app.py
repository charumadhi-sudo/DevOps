from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import json
import os

# Base directory for relative paths
base_dir = os.path.dirname(__file__)

# Load model and scaler
model_path = os.path.join(base_dir, "model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")
metrics_path = os.path.join(base_dir, "metrics.json")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load metrics
with open(metrics_path, "r") as f:
    model_metrics = json.load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Boston Housing Regression API - Lab 5",
    description="MLOps API for predicting house prices using the Boston dataset",
    version="1.0"
)

# Configure Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

class HouseFeatures(BaseModel):
    features: list[float]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Render and serve the index.html front-end
    return templates.TemplateResponse(request, "index.html")

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
