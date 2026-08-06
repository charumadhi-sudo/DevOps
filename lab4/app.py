from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Create app
app = FastAPI(
    title="Boston Housing API",
    description="API for predicting house prices",
    version="1.0"
)

# Feature schema (expects 13 numbers)
class HouseFeatures(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {
        "message": "Boston Housing API is running!",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: HouseFeatures):
    # check features size
    if len(data.features) != 13:
        return {"error": f"Requires exactly 13 features, got {len(data.features)}"}
    
    # scale and predict
    scaled = scaler.transform([data.features])
    price = model.predict(scaled)[0]
    
    return {
        "predicted_price": round(float(price), 4)
    }
