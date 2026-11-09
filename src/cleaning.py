"""
Data Cleaning Module for Store Sales - Time Series Forecasting.
Handles missing oil interpolation, full calendar reindexing (Christmas handling),
transferred holiday resolution, and store active status detection.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np


def clean_oil_prices(oil_df: pd.DataFrame, start_date: str = "2013-01-01", end_date: str = "2017-08-31") -> pd.DataFrame:
    """
    Cleans crude oil price data by establishing a continuous daily calendar
    and filling missing weekend/holiday prices via linear interpolation.
    
    Why this is crucial:
    Crude oil markets (WTI) are closed on weekends and US/international bank holidays.
    Ecuador's economy is heavily oil-dependent, so oil is a key macroeconomic predictor.
    Linear interpolation between Friday's close and Monday's open accurately reflects
    smooth price continuity without artificial step jumps.
    """
    oil = oil_df.copy()
    oil["date"] = pd.to_datetime(oil["date"])
    
    # Create complete calendar covering both historical train and test horizon
    full_calendar = pd.date_range(start=start_date, end=end_date, freq="D", name="date")
    oil_clean = oil.set_index("date").reindex(full_calendar)
    
    # Linear interpolation for gaps, with backward/forward fill for calendar boundaries
    oil_clean["dcoilwtico"] = (
        oil_clean["dcoilwtico"]
        .interpolate(method="linear")
        .bfill()
        .ffill()
        .astype(np.float32)
    )
    
    return oil_clean.reset_index()


def reindex_full_calendar(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindexes the training data to a complete daily grid for every (store_nbr, family) pair.
    
    Why this is crucial:
    Every year on December 25th (Christmas), all Corporación Favorita stores are closed.
    Because sales are 0 on Christmas, the raw Kaggle dataset omitted Dec 25 entirely.
    If dates are omitted, temporal lag structures break! A 'lag 1' on Dec 26 would
    mistakenly grab Dec 24 (2 days prior).
    
    This function creates a complete grid across all 54 stores and 33 product families
    (1,782 distinct time series), filling missing dates with sales=0 and onpromotion=0.
    """
    df = train_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    stores = df["store_nbr"].unique()
    families = df["family"].unique()
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    
    # Generate full Cartesian product: Date x Store x Family
    full_grid = pd.MultiIndex.from_product(
        [dates, stores, families], 
        names=["date", "store_nbr", "family"]
    ).to_frame().reset_index(drop=True)
    
    # Merge existing sales data
    merged = pd.merge(
        full_grid, 
        df[["date", "store_nbr", "family", "sales", "onpromotion", "id"]], 
        on=["date", "store_nbr", "family"], 
        how="left"
    )
    
    # Missing dates (e.g. Dec 25) were days stores were closed: sales = 0, promotions = 0
    merged["sales"] = merged["sales"].fillna(0.0).astype(np.float32)
    merged["onpromotion"] = merged["onpromotion"].fillna(0).astype(np.int16)
    merged["store_nbr"] = merged["store_nbr"].astype(np.int8)
    
    return merged


def clean_holidays(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans holiday events by resolving transferred holidays and categorizing scope.
    
    Key Domain Rules:
    1. Transferred Holidays: When a holiday has `transferred == True`, the celebration
       was officially moved to another date (recorded with type 'Transfer').
       If we don't remove transferred days, we flag a regular working day as a holiday!
    2. Scope hierarchy:
       - National: impacts all stores in Ecuador.
       - Regional: impacts only stores where store.state == locale_name.
       - Local: impacts only stores where store.city == locale_name.
    3. Earthquake relief: April 2016 Manabí earthquake events are flagged.
    """
    holidays = holidays_df.copy()
    holidays["date"] = pd.to_datetime(holidays["date"])
    
    # Discard holidays that were transferred to another date
    active_holidays = holidays[~holidays["transferred"]].copy()
    
    # Deduplicate multiple holidays falling on the same date (e.g. Local + National)
    # Priority: National > Regional > Local
    locale_priority = {"National": 3, "Regional": 2, "Local": 1}
    active_holidays["priority"] = active_holidays["locale"].map(locale_priority).fillna(0)
    
    active_holidays = (
        active_holidays.sort_values(["date", "priority"], ascending=[True, False])
        .drop_duplicates(subset=["date", "locale", "locale_name"])
        .drop(columns=["priority"])
    )
    
    # Flag earthquake events specifically (Apr 16, 2016 and subsequent relief efforts)
    active_holidays["is_earthquake_event"] = active_holidays["description"].str.contains(
        "Terremoto|Manabi", case=False, na=False
    ).astype(np.int8)
    
    return active_holidays


def detect_store_active_dates(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the first date each store recorded non-zero sales.
    
    Why this is crucial:
    Store 52 opened in April 2017, and Store 20 opened in 2015. If we retain years
    of 0 sales before a store actually opened, long rolling features (e.g. 28-day means)
    will be heavily distorted. This function returns the active inception date for each store.
    """
    store_sales = (
        train_df[train_df["sales"] > 0]
        .groupby("store_nbr")["date"]
        .min()
        .reset_index()
        .rename(columns={"date": "first_active_date"})
    )
    return store_sales


def merge_datasets(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    clean_oil_df: pd.DataFrame,
    clean_holidays_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Unified merger joining store metadata, clean oil, and holiday indicators.
    
    Ensures strict adherence to time series integrity without lookahead bias.
    Note: Transactions are deliberately excluded from test-set dependencies
    because transaction counts are not available during the 16-day forecast window.
    """
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Merge Store Metadata (city, state, type, cluster)
    df = pd.merge(df, stores_df, on="store_nbr", how="left")
    
    # 2. Merge Clean Oil Prices
    df = pd.merge(df, clean_oil_df, on="date", how="left")
    
    # 3. National Holidays
    nat_holidays = clean_holidays_df[clean_holidays_df["locale"] == "National"][["date", "description", "is_earthquake_event"]].copy()
    nat_holidays = nat_holidays.drop_duplicates(subset=["date"])
    nat_holidays.rename(columns={"description": "national_holiday_name"}, inplace=True)
    df = pd.merge(df, nat_holidays, on="date", how="left")
    df["is_national_holiday"] = df["national_holiday_name"].notnull().astype(np.int8)
    df["is_earthquake_event"] = df["is_earthquake_event"].fillna(0).astype(np.int8)
    
    # 4. Regional Holidays (match state)
    reg_holidays = clean_holidays_df[clean_holidays_df["locale"] == "Regional"][["date", "locale_name"]].copy()
    reg_holidays.rename(columns={"locale_name": "state"}, inplace=True)
    reg_holidays["is_regional_holiday"] = 1
    reg_holidays = reg_holidays.drop_duplicates(subset=["date", "state"])
    df = pd.merge(df, reg_holidays, on=["date", "state"], how="left")
    df["is_regional_holiday"] = df["is_regional_holiday"].fillna(0).astype(np.int8)
    
    # 5. Local Holidays (match city)
    loc_holidays = clean_holidays_df[clean_holidays_df["locale"] == "Local"][["date", "locale_name"]].copy()
    loc_holidays.rename(columns={"locale_name": "city"}, inplace=True)
    loc_holidays["is_local_holiday"] = 1
    loc_holidays = loc_holidays.drop_duplicates(subset=["date", "city"])
    df = pd.merge(df, loc_holidays, on=["date", "city"], how="left")
    df["is_local_holiday"] = df["is_local_holiday"].fillna(0).astype(np.int8)
    
    # Combined any holiday flag
    df["is_holiday"] = (
        (df["is_national_holiday"] == 1) | 
        (df["is_regional_holiday"] == 1) | 
        (df["is_local_holiday"] == 1)
    ).astype(np.int8)
    
    # Drop verbose text column
    df.drop(columns=["national_holiday_name"], inplace=True, errors="ignore")
    
    return df
