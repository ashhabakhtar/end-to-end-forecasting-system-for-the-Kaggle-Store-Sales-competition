"""
Unit test suite for Step 7 (Baselines) and Step 8 (Multivariate Models).
Validates baseline scoring, Ridge linear regression, Random Forest, and feature importance plotting.
"""

import numpy as np
import pandas as pd
from src.models import (
    evaluate_naive_baselines,
    train_ridge,
    train_random_forest,
    plot_feature_importance
)


def test_models_pipeline():
    # Build synthetic train/val feature sets
    np.random.seed(42)
    N_tr = 500
    N_va = 100
    
    feature_cols = ["lag_16", "lag_21", "rolling_mean_7", "onpromotion", "day_of_week"]
    
    X_tr = pd.DataFrame(np.random.rand(N_tr, len(feature_cols)) * 100, columns=feature_cols)
    y_tr = X_tr["lag_16"] * 0.5 + X_tr["lag_21"] * 0.3 + np.random.normal(0, 5, N_tr)
    y_tr = np.clip(y_tr, 0, None)
    
    X_va = pd.DataFrame(np.random.rand(N_va, len(feature_cols)) * 100, columns=feature_cols)
    y_va = X_va["lag_16"] * 0.5 + X_va["lag_21"] * 0.3 + np.random.normal(0, 5, N_va)
    y_va = np.clip(y_va, 0, None)
    
    # 1. Test Naive Baselines
    val_df_mock = X_va.copy()
    val_df_mock["sales"] = y_va
    base_scores = evaluate_naive_baselines(X_tr, val_df_mock)
    assert "Naive (Lag-16)" in base_scores
    assert "Seasonal Naive (Lag-21)" in base_scores
    print("[PASS] evaluate_naive_baselines passed!")
    
    # 2. Test Ridge Regression
    rf_model, val_preds_rg, rmsle_rg, _ = train_ridge(
        X_tr, np.log1p(y_tr), X_va, np.log1p(y_va), feature_cols
    )
    assert len(val_preds_rg) == N_va
    assert rmsle_rg >= 0.0
    print("[PASS] train_ridge passed!")
    
    # 3. Test Random Forest
    rf_mod, val_preds_rf, rmsle_rf, _ = train_random_forest(
        X_tr, np.log1p(y_tr), X_va, np.log1p(y_va), feature_cols, sample_size=N_tr
    )
    assert len(val_preds_rf) == N_va
    assert rmsle_rf >= 0.0
    print("[PASS] train_random_forest passed!")
    
    # 4. Test Feature Importance Plot
    fig, fi_df = plot_feature_importance(
        rf_mod, feature_cols, top_n=5, save_path="outputs/figures/test_feature_importance.png"
    )
    assert len(fi_df) == 5
    assert "Importance" in fi_df.columns
    print("[PASS] plot_feature_importance passed!")
    
    print("\n>>> ALL STEP 7 & STEP 8 MODEL TESTS PASSED! <<<")


if __name__ == "__main__":
    test_models_pipeline()
