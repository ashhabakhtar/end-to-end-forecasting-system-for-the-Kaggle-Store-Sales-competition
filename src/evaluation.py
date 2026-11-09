"""
Evaluation and Validation Module for Store Sales - Time Series Forecasting.
Provides time-based validation splitters, strict out-of-time evaluation,
and Root Mean Squared Logarithmic Error (RMSLE) metrics.
"""

from typing import Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Root Mean Squared Logarithmic Error (RMSLE):
    RMSLE = sqrt( mean( (ln(1 + max(0, y_pred)) - ln(1 + y_true))^2 ) )
    
    Guarantees negative predictions are clipped to 0 prior to log computation.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Clip negative predictions to zero
    y_pred_clipped = np.clip(y_pred, 0, None)
    
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred_clipped)
    
    rmsle = np.sqrt(np.mean((log_pred - log_true) ** 2))
    return float(rmsle)


def get_train_val_split(
    df: pd.DataFrame, 
    val_days: int = 16,
    date_col: str = "date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a strict out-of-time holdout validation split.
    
    Holds out the final `val_days` (default 16 days) of historical data to mirror
    the exact 16-day forecast horizon of the competition test set.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    max_date = df[date_col].max()
    val_start_date = max_date - pd.Timedelta(days=val_days - 1)
    
    train_mask = df[date_col] < val_start_date
    val_mask = df[date_col] >= val_start_date
    
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    
    print(f"[Validation] Holdout Split Configured ({val_days} Days Horizon):")
    print(f"  -> Train: {train_df[date_col].min().strftime('%Y-%m-%d')} to {train_df[date_col].max().strftime('%Y-%m-%d')} ({len(train_df):,} rows)")
    print(f"  -> Val:   {val_df[date_col].min().strftime('%Y-%m-%d')} to {val_df[date_col].max().strftime('%Y-%m-%d')} ({len(val_df):,} rows)")
    
    return train_df, val_df


def get_rolling_time_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    val_days: int = 16,
    date_col: str = "date"
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, str], None, None]:
    """
    Generates rolling-origin expanding window time series cross-validation folds.
    
    Fold k trains on historical data up to T - (k * val_days) and validates
    on the subsequent val_days window.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max()
    
    for k in reversed(range(n_splits)):
        fold_val_end = max_date - pd.Timedelta(days=k * val_days)
        fold_val_start = fold_val_end - pd.Timedelta(days=val_days - 1)
        
        train_fold = df[df[date_col] < fold_val_start].reset_index(drop=True)
        val_fold = df[(df[date_col] >= fold_val_start) & (df[date_col] <= fold_val_end)].reset_index(drop=True)
        
        fold_name = f"Fold_{n_splits - k} (Val: {fold_val_start.strftime('%m/%d')} to {fold_val_end.strftime('%m/%d')})"
        yield train_fold, val_fold, fold_name


def scale_numeric_features(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales numerical features using StandardScaler.
    
    CRITICAL ANTI-LEAKAGE RULE:
    The scaler is fit ONLY on X_train. It transforms both X_train and X_val.
    This is required for regularized linear models (Ridge/Lasso) and neural nets.
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    # Fit strictly on train
    scaler.fit(X_train[numeric_cols])
    
    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    
    return X_train_scaled, X_val_scaled, scaler
