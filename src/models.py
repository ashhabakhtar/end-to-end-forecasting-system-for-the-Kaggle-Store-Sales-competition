"""
Modeling Module for Store Sales - Time Series Forecasting.
Provides modular implementations for:
- Naive & Seasonal Naive Baselines
- LightGBM, XGBoost, Random Forest, and Ridge Regressors
- Feature Importance extraction and visualization
- Model comparison tables
"""

from typing import Dict, List, Optional, Tuple, Union
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from src.evaluation import calculate_rmsle


def evaluate_naive_baselines(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Computes benchmark scores for:
    1. Naive Lag-16 Baseline: predicts sales using the most recent known lag (lag_16).
    2. Seasonal Naive Baseline: predicts sales using the same weekday from 3 weeks prior (lag_21).
    """
    # 1. Pure Lag-16 Naive
    val_preds_naive = val_df["lag_16"].fillna(0.0).values
    rmsle_naive = calculate_rmsle(val_df["sales"].values, val_preds_naive)
    
    # 2. Seasonal Naive (lag_21 matching same day of week)
    val_preds_snaive = val_df["lag_21"].fillna(val_df["lag_16"]).fillna(0.0).values
    rmsle_snaive = calculate_rmsle(val_df["sales"].values, val_preds_snaive)
    
    print(f"[Baseline 1] Naive (Lag-16) Validation RMSLE:          {rmsle_naive:.4f}")
    print(f"[Baseline 2] Seasonal Naive (Lag-21) Validation RMSLE:   {rmsle_snaive:.4f}")
    
    return {
        "Naive (Lag-16)": round(rmsle_naive, 4),
        "Seasonal Naive (Lag-21)": round(rmsle_snaive, 4)
    }


def train_lightgbm(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_val: pd.DataFrame, 
    y_val: np.ndarray,
    feature_names: List[str],
    cat_features: Optional[List[str]] = None,
    params: Optional[Dict] = None
) -> Tuple[object, np.ndarray, float, float]:
    """
    Trains a LightGBM Regressor on log-transformed targets.
    Returns: (fitted_model, val_predictions_real_scale, val_rmsle, elapsed_seconds)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is not installed. Please install via: pip install lightgbm")
        
    default_params = {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    if params:
        default_params.update(params)
        
    start_time = time.time()
    
    # Ensure categorical features are category dtype
    X_tr = X_train[feature_names].copy()
    X_va = X_val[feature_names].copy()
    if cat_features:
        for c in cat_features:
            if c in X_tr.columns:
                X_tr[c] = X_tr[c].astype("category")
                X_va[c] = X_va[c].astype("category")
                
    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_tr, y_train)
    elapsed = time.time() - start_time
    
    # Predict in log scale and invert back with expm1
    val_pred_log = model.predict(X_va)
    val_pred_real = np.clip(np.expm1(val_pred_log), 0, None)
    
    # Invert validation ground truth for RMSLE calculation
    y_val_real = np.expm1(y_val) if np.min(y_val) >= 0 else y_val
    rmsle = calculate_rmsle(y_val_real, val_pred_real)
    
    print(f"[LightGBM] Train completed in {elapsed:.1f}s | Val RMSLE: {rmsle:.4f}")
    return model, val_pred_real, rmsle, elapsed


def train_xgboost(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_val: pd.DataFrame, 
    y_val: np.ndarray,
    feature_names: List[str],
    params: Optional[Dict] = None
) -> Tuple[object, np.ndarray, float, float]:
    """
    Trains an XGBoost Regressor on log-transformed targets.
    Returns: (fitted_model, val_predictions_real_scale, val_rmsle, elapsed_seconds)
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost is not installed. Please install via: pip install xgboost")
        
    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0
    }
    if params:
        default_params.update(params)
        
    start_time = time.time()
    
    X_tr = X_train[feature_names].select_dtypes(include=[np.number])
    X_va = X_val[feature_names].select_dtypes(include=[np.number])
    
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_tr, y_train)
    elapsed = time.time() - start_time
    
    val_pred_log = model.predict(X_va)
    val_pred_real = np.clip(np.expm1(val_pred_log), 0, None)
    
    y_val_real = np.expm1(y_val) if np.min(y_val) >= 0 else y_val
    rmsle = calculate_rmsle(y_val_real, val_pred_real)
    
    print(f"[XGBoost] Train completed in {elapsed:.1f}s | Val RMSLE: {rmsle:.4f}")
    return model, val_pred_real, rmsle, elapsed


def train_random_forest(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_val: pd.DataFrame, 
    y_val: np.ndarray,
    feature_names: List[str],
    sample_size: Optional[int] = 100000
) -> Tuple[object, np.ndarray, float, float]:
    """
    Trains a Random Forest Regressor. Uses subsampling on training data
    to maintain fast CPU execution on Kaggle without sacrificing tree diversity.
    """
    start_time = time.time()
    
    X_tr = X_train[feature_names].select_dtypes(include=[np.number]).fillna(0)
    X_va = X_val[feature_names].select_dtypes(include=[np.number]).fillna(0)
    
    # Subsample for CPU efficiency if dataset is massive
    if sample_size and len(X_tr) > sample_size:
        idx = np.random.choice(len(X_tr), size=sample_size, replace=False)
        X_tr = X_tr.iloc[idx]
        y_tr = y_train[idx]
    else:
        y_tr = y_train
        
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=12, 
        min_samples_split=10,
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    elapsed = time.time() - start_time
    
    val_pred_log = model.predict(X_va)
    val_pred_real = np.clip(np.expm1(val_pred_log), 0, None)
    
    y_val_real = np.expm1(y_val) if np.min(y_val) >= 0 else y_val
    rmsle = calculate_rmsle(y_val_real, val_pred_real)
    
    print(f"[RandomForest] Train completed in {elapsed:.1f}s | Val RMSLE: {rmsle:.4f}")
    return model, val_pred_real, rmsle, elapsed


def train_ridge(
    X_train_scaled: pd.DataFrame, 
    y_train: np.ndarray, 
    X_val_scaled: pd.DataFrame, 
    y_val: np.ndarray,
    feature_names: List[str],
    alpha: float = 10.0
) -> Tuple[object, np.ndarray, float, float]:
    """
    Trains an L2 Regularized Ridge Regression model on standardized features.
    """
    start_time = time.time()
    
    X_tr = X_train_scaled[feature_names].select_dtypes(include=[np.number]).fillna(0)
    X_va = X_val_scaled[feature_names].select_dtypes(include=[np.number]).fillna(0)
    
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_tr, y_train)
    elapsed = time.time() - start_time
    
    val_pred_log = model.predict(X_va)
    val_pred_real = np.clip(np.expm1(val_pred_log), 0, None)
    
    y_val_real = np.expm1(y_val) if np.min(y_val) >= 0 else y_val
    rmsle = calculate_rmsle(y_val_real, val_pred_real)
    
    print(f"[Ridge Regression] Train completed in {elapsed:.1f}s | Val RMSLE: {rmsle:.4f}")
    return model, val_pred_real, rmsle, elapsed


def plot_feature_importance(
    model: object, 
    feature_names: List[str], 
    top_n: int = 10,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Extracts and visualizes the Top N most influential features from a tree model.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute.")
        
    fi_df = pd.DataFrame({
        "Feature": feature_names[:len(importances)],
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    
    top_fi = fi_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_fi, x="Importance", y="Feature", hue="Feature", legend=False, ax=ax, palette="Blues_r")
    ax.set_title(f"Top {top_n} Most Influential Features (LightGBM Gini Gain)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Importance (Split Gain)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
    return fig, fi_df
