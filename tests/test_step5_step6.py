"""
Unit test suite for Step 5 (Scaling & Target Transformation) and Step 6 (Validation Strategy).
Validates RMSLE calculation, negative value clipping, temporal train/val split boundaries,
and train-only StandardScaler fitting.
"""

import numpy as np
import pandas as pd
from src.evaluation import (
    calculate_rmsle,
    get_train_val_split,
    get_rolling_time_splits,
    scale_numeric_features
)


def test_rmsle_calculation():
    # Identical predictions should have 0.0 error
    y_true = np.array([10.0, 50.0, 100.0, 0.0])
    y_pred = np.array([10.0, 50.0, 100.0, 0.0])
    assert calculate_rmsle(y_true, y_pred) == 0.0, "RMSLE for identical arrays must be 0!"
    
    # Negative predictions must be clipped to zero without raising log errors
    y_pred_neg = np.array([-5.0, 50.0, 100.0, 0.0])
    err = calculate_rmsle(y_true, y_pred_neg)
    assert not np.isnan(err), "Negative predictions must not cause NaN in RMSLE"
    assert err > 0.0
    print("[PASS] test_rmsle_calculation passed!")


def test_time_based_split():
    # Build 60 days of data
    dates = pd.date_range("2017-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "sales": np.arange(60)
    })
    
    train_df, val_df = get_train_val_split(df, val_days=16)
    
    assert len(val_df) == 16, f"Validation set must have exactly 16 days, got {len(val_df)}"
    assert len(train_df) == 44, f"Train set must have 44 days, got {len(train_df)}"
    assert train_df["date"].max() < val_df["date"].min(), "Train dates must strictly precede validation dates!"
    print("[PASS] test_time_based_split passed!")


def test_feature_scaling():
    # Test scaler fits strictly on train
    X_train = pd.DataFrame({"sales_lag": [10.0, 20.0, 30.0, 40.0, 50.0]})
    X_val = pd.DataFrame({"sales_lag": [60.0, 70.0]})
    
    X_tr_sc, X_va_sc, scaler = scale_numeric_features(X_train, X_val, ["sales_lag"])
    
    # Train mean should be 0, std should be 1
    np.testing.assert_almost_equal(X_tr_sc["sales_lag"].mean(), 0.0, decimal=5)
    np.testing.assert_almost_equal(X_tr_sc["sales_lag"].std(ddof=0), 1.0, decimal=5)
    
    # Val values should be scaled using train mean (30) and std (~14.14)
    expected_val_0 = (60.0 - 30.0) / scaler.scale_[0]
    np.testing.assert_almost_equal(X_va_sc["sales_lag"].iloc[0], expected_val_0, decimal=5)
    print("[PASS] test_feature_scaling passed!")


if __name__ == "__main__":
    test_rmsle_calculation()
    test_time_based_split()
    test_feature_scaling()
    print("\n>>> ALL STEP 5 & STEP 6 TESTS PASSED! <<<")
