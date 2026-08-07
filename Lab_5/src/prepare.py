import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def download_data(url: str) -> pd.DataFrame:
    print(f"[prepare] Downloading raw data from {url}")
    return pd.read_csv(url)

def preprocess_and_split(df: pd.DataFrame, params: dict):
    # Clean column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    
    # Handle duplicates and nulls
    df = df.drop_duplicates()
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
        
    # Rename target column 'medv' to 'target'
    if "medv" in df.columns:
        df = df.rename(columns={"medv": "target"})
    
    df["target"] = df["target"].astype(float)
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Split
    fe_params = params["feature_engineering"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=fe_params["test_size"], 
        random_state=fe_params["random_state"]
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    
    # Recombine features and targets
    train_df = X_train_scaled.copy()
    train_df["target"] = y_train.values
    
    test_df = X_test_scaled.copy()
    test_df["target"] = y_test.values
    
    return train_df, test_df, scaler

def save_data_and_scaler(train_df, test_df, scaler):
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    joblib.dump(scaler, "model/scaler.pkl")
    
    print("[prepare] Successfully split data and saved scaler.pkl")

def main():
    params = load_params()
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = download_data(url)
    train_df, test_df, scaler = preprocess_and_split(df, params)
    save_data_and_scaler(train_df, test_df, scaler)

if __name__ == "__main__":
    main()
