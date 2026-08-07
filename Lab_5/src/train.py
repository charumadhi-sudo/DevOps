import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_train_data(path: str = "data/train.csv") -> pd.DataFrame:
    print(f"[train] Loading training data from {path}")
    return pd.read_csv(path)

def train_model(df: pd.DataFrame, n_estimators: int, max_depth: int, random_state: int) -> RandomForestRegressor:
    X_train = df.drop(columns=["target"])
    y_train = df["target"]
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("[train] Model training successfully completed.")
    return model

def save_model(model, out_dir: str = "model"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.pkl")
    joblib.dump(model, out_path)
    print(f"[train] Model saved -> {out_path}")

def main():
    params = load_params()["model_building"]
    df = load_train_data()
    model = train_model(
        df,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    save_model(model)

if __name__ == "__main__":
    main()
