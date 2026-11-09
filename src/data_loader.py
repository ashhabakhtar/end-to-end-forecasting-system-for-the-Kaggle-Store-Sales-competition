"""
Data Loader Module for Store Sales - Time Series Forecasting.
Handles robust file ingestion, path resolution (local vs Kaggle),
and memory-optimized schema casting.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import os
import pandas as pd
import numpy as np


def resolve_data_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Automatically detects whether the code is running on Kaggle or a local machine.
    
    Priority:
    1. custom_path if provided and exists
    2. Kaggle environment: /kaggle/input/store-sales-time-series-forecasting
    3. Local directory: ./data
    4. Current directory: .
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    kaggle_path = Path("/kaggle/input/store-sales-time-series-forecasting")
    if kaggle_path.exists():
        return kaggle_path

    local_data_path = Path(__file__).resolve().parent.parent / "data"
    if local_data_path.exists():
        return local_data_path

    return Path(".")


def optimize_dtypes(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Reduces memory consumption by downcasting default 64-bit numeric types
    to smaller representations (int8, int16, float32) without losing precision.
    
    Why this matters:
    The train set contains over 3 million rows. In default 64-bit pandas,
    this takes >1.5 GB RAM. Downcasting reduces RAM to ~350 MB, preventing OOM
    crashes on Kaggle CPU notebooks.
    """
    df = df.copy()

    if "store_nbr" in df.columns:
        df["store_nbr"] = df["store_nbr"].astype(np.int8)

    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(np.int8)

    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].astype(np.int16)

    if "sales" in df.columns:
        df["sales"] = df["sales"].astype(np.float32)

    if "dcoilwtico" in df.columns:
        df["dcoilwtico"] = df["dcoilwtico"].astype(np.float32)

    if "transactions" in df.columns:
        df["transactions"] = df["transactions"].astype(np.int16)

    return df


def load_raw_data(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all Kaggle Store Sales CSV files with parsed dates and optimized datatypes.
    
    Returns:
        dict: A dictionary containing:
            - 'train': Training sales data (2013-01-01 to 2017-08-15)
            - 'test': Prediction targets (2017-08-16 to 2017-08-31)
            - 'stores': Store metadata (city, state, type, cluster)
            - 'oil': Daily crude oil price (WTI)
            - 'holidays': Holiday calendar (local, regional, national)
            - 'transactions': Daily customer transactions per store
            - 'sample_sub': Submission format template
    """
    path = resolve_data_path(data_dir)
    print(f"[DataLoader] Ingesting files from: {path}")

    files = {
        "train": ("train.csv", ["date"]),
        "test": ("test.csv", ["date"]),
        "stores": ("stores.csv", None),
        "oil": ("oil.csv", ["date"]),
        "holidays": ("holidays_events.csv", ["date"]),
        "transactions": ("transactions.csv", ["date"]),
        "sample_sub": ("sample_submission.csv", None),
    }

    data: Dict[str, pd.DataFrame] = {}

    for key, (filename, parse_dates) in files.items():
        file_path = path / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing expected dataset file: {file_path}. "
                f"Please ensure '{filename}' is placed in {path}."
            )
        
        df = pd.read_csv(file_path, parse_dates=parse_dates)
        df = optimize_dtypes(df, key)
        data[key] = df
        print(f"  [PASS] Loaded {key:12s} | Shape: {df.shape} | RAM: {df.memory_usage().sum() / 1e6:.2f} MB")

    return data


def summarize_dataset(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generates a high-level summary table of all loaded datasets.
    """
    summary = []
    for name, df in data.items():
        date_col = "date" if "date" in df.columns else None
        date_range = (
            f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            if date_col
            else "N/A"
        )
        summary.append({
            "Dataset": name,
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Date Range": date_range,
            "Missing Values": df.isnull().sum().sum(),
            "Memory (MB)": round(df.memory_usage().sum() / 1e6, 2),
        })
    return pd.DataFrame(summary)
