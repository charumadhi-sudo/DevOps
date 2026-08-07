import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    """Verify that the home page renders and returns HTML response."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Boston Housing Price Predictor" in response.text

def test_health():
    """Verify that the health check endpoint returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_metrics():
    """Verify that the metrics endpoint returns correct metrics fields."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "mae" in data
    assert "mse" in data
    assert "r2_score" in data
    assert data["r2_score"] == 0.73

def test_predict_success():
    """Verify that prediction succeeds with 13 valid features."""
    payload = {
        "features": [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296, 15.3, 396.9, 4.98]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0

def test_predict_invalid_feature_count():
    """Verify that predicting with incorrect number of features returns error."""
    payload = {
        "features": [0.00632, 18.0, 2.31] # Only 3 features instead of 13
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "expects exactly 13 features" in data["error"]
