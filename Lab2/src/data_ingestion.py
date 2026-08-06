"""
Stage 1: Data Ingestion (Regression Version)
------------------------
Loads the Boston Housing dataset from a remote CSV and dumps it as a raw CSV file.

Output:
    data/raw/data.csv
"""

import os
import pandas as pd


def load_data() -> pd.DataFrame:
    """Load the Boston housing dataset into a DataFrame."""
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    print(f"[data_ingestion] Downloading data from {url}")
    df = pd.read_csv(url)
    return df


def save_raw_data(df: pd.DataFrame, out_dir: str = "data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_ingestion] Saved raw data -> {out_path} (shape={df.shape})")


def main():
    df = load_data()
    save_raw_data(df)


if __name__ == "__main__":
    main()
