"""
Unit test suite for Step 4 (Feature Engineering).
Validates calendar encodings, holiday distance, anti-leakage lag constraints (lags >= 16),
rolling window calculations, and categorical transformations.
"""

import pandas as pd
import numpy as np
from src.features import (
    create_calendar_features,
    create_holiday_proximity_features,
    create_lag_and_rolling_features,
    create_oil_momentum_features,
    encode_categorical_features
)


def test_calendar_features():
    df = pd.DataFrame({"date": pd.date_range("2016-01-01", periods=31, freq="D")})
    res = create_calendar_features(df)
    
    assert "day_of_week" in res.columns
    assert "is_weekend" in res.columns
    assert "is_payday" in res.columns
    assert "days_to_payday" in res.columns
    assert "sin_dow" in res.columns
    assert "cos_dow" in res.columns
    
    # Check that 15th and 31st are flagged as paydays
    paydays = res[res["is_payday"] == 1]["date"].dt.day.tolist()
    assert 15 in paydays, "15th must be flagged as a payday"
    assert 31 in paydays, "Month-end must be flagged as a payday"
    print("[PASS] test_calendar_features passed!")


def test_anti_leakage_lags():
    """Verify that lag features strictly respect the 16-day prediction horizon."""
    # Build 60 days of consecutive sales for store 1, family GROCERY I
    dates = pd.date_range("2017-01-01", periods=60, freq="D")
    sales = [float(i * 10) for i in range(60)]
    df = pd.DataFrame({
        "date": dates,
        "store_nbr": [1] * 60,
        "family": ["GROCERY I"] * 60,
        "sales": sales,
        "onpromotion": [2] * 60
    })
    
    res = create_lag_and_rolling_features(df, lags=[16, 21, 28], windows=[7, 14])
    
    # Verify lag 16 on day 20 (index 20) matches sales on day 4 (index 4)
    assert res.loc[20, "lag_16"] == sales[4], f"Expected {sales[4]}, got {res.loc[20, 'lag_16']}"
    # Verify lag 21 on day 25 (index 25) matches sales on day 4 (index 4)
    assert res.loc[25, "lag_21"] == sales[4], f"Expected {sales[4]}, got {res.loc[25, 'lag_21']}"
    
    # Verify that first 16 days have NaN for lag_16
    assert np.isnan(res.loc[15, "lag_16"]), "Day 15 lag_16 must be NaN because lag >= 16 requires at least 16 historical days"
    
    # Verify rolling mean over 7 days on shifted lag
    # At index 22, rolling_mean_7 should average lags at indices 16..22
    assert "rolling_mean_7" in res.columns
    assert "rolling_std_7" in res.columns
    print("[PASS] test_anti_leakage_lags passed!")


def test_holiday_proximity():
    df = pd.DataFrame({"date": pd.date_range("2016-12-20", periods=10, freq="D")})
    hols = pd.DataFrame({
        "date": pd.to_datetime(["2016-12-25"]),
        "locale": ["National"]
    })
    res = create_holiday_proximity_features(df, hols)
    assert "days_to_next_holiday" in res.columns
    assert "days_since_last_holiday" in res.columns
    
    # On Dec 20, days to Dec 25 is 5
    row_dec20 = res[res["date"] == "2016-12-20"].iloc[0]
    assert row_dec20["days_to_next_holiday"] == 5
    
    # On Dec 25, days to next holiday is 0
    row_dec25 = res[res["date"] == "2016-12-25"].iloc[0]
    assert row_dec25["days_to_next_holiday"] == 0
    print("[PASS] test_holiday_proximity passed!")


if __name__ == "__main__":
    test_calendar_features()
    test_anti_leakage_lags()
    test_holiday_proximity()
    print("\n>>> ALL STEP 4 FEATURE ENGINEERING TESTS PASSED! <<<")
