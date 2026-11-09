"""
Unit test suite for Step 3 (Exploratory Data Analysis).
Verifies that all EDA functions execute cleanly on simulated retail data.
"""

import pandas as pd
import numpy as np
from src.eda import (
    plot_sales_over_time,
    decompose_time_series,
    analyze_calendar_and_paydays,
    analyze_product_hierarchy,
    plot_autocorrelation,
    check_stationarity_adf
)


def create_synthetic_sales_data():
    """Generates 120 days of realistic synthetic sales data across 3 stores and 2 families."""
    dates = pd.date_range("2016-01-01", periods=120, freq="D")
    records = []
    for d in dates:
        # Weekend lift (Saturday=5, Sunday=6)
        is_weekend = d.dayofweek in [5, 6]
        base = 200 + (80 if is_weekend else 0) + (d.day % 15 == 0) * 120
        for s in [1, 2]:
            for f in ["GROCERY I", "BEVERAGES"]:
                noise = np.random.normal(0, 10)
                sales = max(0, base + noise)
                records.append({
                    "date": d,
                    "store_nbr": s,
                    "family": f,
                    "sales": sales,
                    "onpromotion": int(np.random.poisson(3))
                })
    return pd.DataFrame(records)


def test_eda_pipeline():
    df = create_synthetic_sales_data()
    
    # 1. Resample over time
    fig1, daily = plot_sales_over_time(df, save_path="outputs/figures/test_sales_over_time.png")
    assert len(daily) == 120, "Daily series must have 120 days"
    print("[PASS] plot_sales_over_time passed!")
    
    # 2. Decomposition
    fig2 = decompose_time_series(daily["sales"], period=7, save_path="outputs/figures/test_decomp.png")
    assert fig2 is not None
    print("[PASS] decompose_time_series passed!")
    
    # 3. Calendar & Payday
    fig3, cal_df = analyze_calendar_and_paydays(df, save_path="outputs/figures/test_calendar.png")
    assert "is_payday" in cal_df.columns
    print("[PASS] analyze_calendar_and_paydays passed!")
    
    # 4. Product Hierarchy Pareto
    fig4, family_sales = analyze_product_hierarchy(df, top_n=2, save_path="outputs/figures/test_hierarchy.png")
    assert len(family_sales) == 2
    print("[PASS] analyze_product_hierarchy passed!")
    
    # 5. Autocorrelation
    fig5 = plot_autocorrelation(daily["sales"], max_lags=28, save_path="outputs/figures/test_acf.png")
    assert fig5 is not None
    print("[PASS] plot_autocorrelation passed!")
    
    # 6. Stationarity ADF test
    adf_res = check_stationarity_adf(daily["sales"], "Daily Sales")
    assert "Stationary" in adf_res
    print("[PASS] check_stationarity_adf passed!")
    
    print("\n>>> ALL STEP 3 EDA TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    test_eda_pipeline()
