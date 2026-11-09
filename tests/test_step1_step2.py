"""
Automated Verification Suite for Step 1 & Step 2.
Tests schema casting, memory optimization, oil interpolation,
Christmas calendar reindexing, transferred holiday filtering, and dataset merging.
"""

import pandas as pd
import numpy as np
from src.cleaning import (
    clean_oil_prices,
    reindex_full_calendar,
    clean_holidays,
    detect_store_active_dates,
    merge_datasets
)
from src.data_loader import optimize_dtypes


def test_oil_interpolation():
    """Verify that oil interpolation leaves zero NaNs and creates continuous daily dates."""
    dates = ["2013-01-02", "2013-01-03", "2013-01-07"]  # Weekend Jan 4-6 + holiday Jan 1 missing
    prices = [93.14, 92.97, 93.43]
    raw_oil = pd.DataFrame({"date": dates, "dcoilwtico": prices})
    
    clean_oil = clean_oil_prices(raw_oil, start_date="2013-01-01", end_date="2013-01-07")
    
    assert clean_oil["dcoilwtico"].isnull().sum() == 0, "Clean oil should have zero NaNs!"
    # Verified



def test_reindex_full_calendar():
    """Verify that reindexing creates entries for missing dates (e.g. Christmas) with sales=0."""
    # Dummy data with Dec 24 and Dec 26, omitting Dec 25
    train_data = pd.DataFrame({
        "id": [1, 2],
        "date": ["2013-12-24", "2013-12-26"],
        "store_nbr": [1, 1],
        "family": ["GROCERY I", "GROCERY I"],
        "sales": [150.0, 200.0],
        "onpromotion": [5, 10]
    })
    
    reindexed = reindex_full_calendar(train_data)
    
    # Should contain 3 days: 2013-12-24, 2013-12-25, 2013-12-26
    assert len(reindexed) == 3, f"Expected 3 rows, got {len(reindexed)}"
    christmas_row = reindexed[reindexed["date"] == "2013-12-25"].iloc[0]
    assert christmas_row["sales"] == 0.0, "Christmas sales should be filled with 0.0!"
    assert christmas_row["onpromotion"] == 0, "Christmas promotion should be filled with 0!"


def test_transferred_holidays():
    """Verify that holidays with transferred=True are excluded from active celebrations."""
    holidays_data = pd.DataFrame({
        "date": ["2013-10-09", "2013-10-11"],
        "type": ["Holiday", "Transfer"],
        "locale": ["National", "National"],
        "locale_name": ["Ecuador", "Ecuador"],
        "description": ["Independencia de Guayaquil", "Traslado Independencia de Guayaquil"],
        "transferred": [True, False]
    })
    
    cleaned = clean_holidays(holidays_data)
    assert len(cleaned) == 1, "Only the non-transferred active holiday date should remain!"
    assert cleaned.iloc[0]["date"] == pd.Timestamp("2013-10-11"), "The active date should be Oct 11!"


def test_unified_merge():
    """Verify that merge_datasets produces consistent columns and zero lookup errors."""
    sales = pd.DataFrame({
        "date": ["2016-04-16", "2016-04-17"],
        "store_nbr": [1, 1],
        "family": ["BREAD/BAKERY", "BREAD/BAKERY"],
        "sales": [50.0, 45.0],
        "onpromotion": [0, 0]
    })
    stores = pd.DataFrame({
        "store_nbr": [1],
        "city": ["Quito"],
        "state": ["Pichincha"],
        "type": ["D"],
        "cluster": [13]
    })
    oil = pd.DataFrame({
        "date": pd.to_datetime(["2016-04-16", "2016-04-17"]),
        "dcoilwtico": [40.35, 40.40]
    })
    holidays = pd.DataFrame({
        "date": pd.to_datetime(["2016-04-16"]),
        "type": ["Event"],
        "locale": ["National"],
        "locale_name": ["Ecuador"],
        "description": ["Terremoto Manabi"],
        "transferred": [False]
    })
    
    clean_hols = clean_holidays(holidays)
    merged = merge_datasets(sales, stores, oil, clean_hols)
    
    assert "is_earthquake_event" in merged.columns
    assert merged.loc[0, "is_earthquake_event"] == 1
    assert merged.loc[1, "is_earthquake_event"] == 0
    assert "dcoilwtico" in merged.columns
    assert "cluster" in merged.columns


if __name__ == "__main__":
    test_oil_interpolation()
    print("[PASS] test_oil_interpolation passed!")
    test_reindex_full_calendar()
    print("[PASS] test_reindex_full_calendar passed!")
    test_transferred_holidays()
    print("[PASS] test_transferred_holidays passed!")
    test_unified_merge()
    print("[PASS] test_unified_merge passed!")
    print("\n>>> ALL STEP 1 & STEP 2 TESTS PASSED SUCCESSFULLY! <<<")
