import json
import joblib
import numpy as np
import pandas as pd
import yaml
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(path: str = "model/model.pkl"):
    print(f"[evaluate] Loading model from {path}")
    return joblib.load(path)

def load_test_data(path: str = "data/test.csv") -> pd.DataFrame:
    print(f"[evaluate] Loading test data from {path}")
    return pd.read_csv(path)

def evaluate(model, df: pd.DataFrame) -> dict:
    X_test = df.drop(columns=["target"])
    y_test = df["target"]
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2)
    }

def save_metrics(metrics: dict, path: str = "metrics.json"):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[evaluate] Saved metrics to {path}:")
    print(json.dumps(metrics, indent=4))

def main():
    params = load_params()
    model = load_model()
    df = load_test_data()
    metrics = evaluate(model, df)
    save_metrics(metrics)
    
    # Quality gate check
    min_r2 = params.get("evaluate", {}).get("min_r2", 0.70)
    actual_r2 = metrics["r2_score"]
    
    print(f"[evaluate] Quality Gate Check: Min R2 Required = {min_r2}, Actual R2 = {actual_r2:.4f}")
    if actual_r2 < min_r2:
        print("[evaluate] ERROR: Model accuracy is below the quality gate threshold!")
        sys.exit(1)
    else:
        print("[evaluate] SUCCESS: Model accuracy passed the quality gate check!")
        sys.exit(0)

if __name__ == "__main__":
    main()
