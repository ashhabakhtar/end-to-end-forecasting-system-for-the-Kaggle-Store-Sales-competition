"""
Ensemble Module for Store Sales - Time Series Forecasting.
Implements:
- BaggingRegressor with LightGBM / DecisionTree base estimators
- Weighted Blending across diverse model families (GBDT + Linear + RF)
- VotingRegressor implementations
- Theoretical variance reduction metrics
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from src.evaluation import calculate_rmsle


def calculate_blend_predictions(
    predictions_dict: Dict[str, np.ndarray],
    weights_dict: Dict[str, float]
) -> np.ndarray:
    """
    Computes a weighted linear combination of predictions from multiple models:
    y_blend = sum( w_m * y_m ) / sum( w_m )
    """
    total_weight = sum(weights_dict.values())
    normalized_weights = {m: w / total_weight for m, w in weights_dict.items()}
    
    first_pred = next(iter(predictions_dict.values()))
    blend = np.zeros_like(first_pred, dtype=np.float64)
    
    for model_name, preds in predictions_dict.items():
        weight = normalized_weights.get(model_name, 0.0)
        blend += weight * preds
        
    return np.clip(blend, 0, None)


def evaluate_ensemble_blend(
    predictions_dict: Dict[str, np.ndarray],
    weights_dict: Dict[str, float],
    y_true_real: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Evaluates a weighted blend on the validation ground truth.
    Returns: (blend_rmsle, blend_predictions)
    """
    blend_preds = calculate_blend_predictions(predictions_dict, weights_dict)
    blend_rmsle = calculate_rmsle(y_true_real, blend_preds)
    
    print("\n" + "="*50)
    print("ENSEMBLE BLEND EVALUATION")
    print("="*50)
    for model_name, w in weights_dict.items():
        print(f"  -> {model_name:20s} Weight: {w:.2f}")
    print(f"\nFinal Ensemble Validation RMSLE: {blend_rmsle:.4f}")
    print("="*50)
    
    return blend_rmsle, blend_preds


def train_voting_ensemble(
    estimators: List[Tuple[str, object]],
    weights: List[float],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray
) -> Tuple[VotingRegressor, np.ndarray, float]:
    """
    Fits a Scikit-Learn VotingRegressor on pre-scaled numerical feature matrices.
    """
    voting = VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1)
    voting.fit(X_train, y_train)
    
    val_pred_log = voting.predict(X_val)
    val_pred_real = np.clip(np.expm1(val_pred_log), 0, None)
    
    y_val_real = np.expm1(y_val) if np.min(y_val) >= 0 else y_val
    rmsle = calculate_rmsle(y_val_real, val_pred_real)
    
    return voting, val_pred_real, rmsle
