"""
Feature Engineering Module for Store Sales - Time Series Forecasting.
Builds leak-free temporal features strictly aligned with the 16-day forecast horizon:
- Calendar & cyclic trigonometric encodings (day of week, year, payday quincena)
- Holiday proximity indicators (days to next / days since last holiday)
- Lag features (lags >= 16: 16, 17, 18, 21, 28, 35)
- Rolling window statistics on shifted lags (mean, std over 7, 14, 28 days)
- Promotion and crude oil momentum features
- Categorical type casting and frequency encoding
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def create_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Generates temporal calendar and cyclic trigonometric features.
    
    Why cyclic encoding:
    Sunday (6) and Monday (0) are adjacent in reality, but numerically 6 and 0
    are far apart. Sine/Cosine encodings project calendar cycles onto a 2D circle,
    preserving continuous cyclic proximity for linear and distance-based models.
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    
    # Standard calendar
    df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
    df["day_of_month"] = dt.dt.day.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    df["year"] = dt.dt.year.astype(np.int16)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(np.int8)
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(np.int8)
    
    # Payday (Quincena) logic: 15th and month-end
    is_15th = dt.dt.day == 15
    is_m_end = dt.dt.is_month_end
    df["is_payday"] = (is_15th | is_m_end).astype(np.int8)
    
    # Proximity to nearest payday (distance in days)
    days_in_month = dt.dt.days_in_month
    dist_15 = (dt.dt.day - 15).abs()
    dist_end = (days_in_month - dt.dt.day).abs()
    dist_start = (dt.dt.day - 1).abs()
    df["days_to_payday"] = np.minimum(np.minimum(dist_15, dist_end), dist_start).astype(np.int8)
    
    # Cyclic Sine/Cosine transformations
    # 1. Day of Week cycle (period 7)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)
    
    # 2. Day of Year cycle (period 365.25)
    day_of_year = dt.dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    
    return df


def create_holiday_proximity_features(
    df: pd.DataFrame, 
    clean_hols_df: pd.DataFrame, 
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Computes days until the next holiday and days elapsed since the last holiday.
    Consumer spending surges prior to major holidays (e.g. Christmas, Carnival, New Year).
    """
    df = df.copy()
    dates = pd.DataFrame({date_col: pd.to_datetime(df[date_col].unique())}).sort_values(date_col)
    
    # Extract distinct national holiday dates
    nat_dates = pd.to_datetime(
        clean_hols_df[clean_hols_df["locale"] == "National"][date_col].unique()
    )
    nat_dates = np.sort(nat_dates)
    
    # Compute proximity for each unique date
    all_dates = dates[date_col].values
    
    days_to_next = []
    days_since_last = []
    
    for d in all_dates:
        future_hols = nat_dates[nat_dates >= d]
        past_hols = nat_dates[nat_dates <= d]
        
        d_next = (future_hols[0] - d).astype("timedelta64[D]").astype(int) if len(future_hols) > 0 else 99
        d_past = (d - past_hols[-1]).astype("timedelta64[D]").astype(int) if len(past_hols) > 0 else 99
        
        days_to_next.append(min(d_next, 99))
        days_since_last.append(min(d_past, 99))
        
    dates["days_to_next_holiday"] = np.array(days_to_next, dtype=np.int8)
    dates["days_since_last_holiday"] = np.array(days_since_last, dtype=np.int8)
    
    return pd.merge(df, dates, on=date_col, how="left")


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    lags: List[int] = [16, 17, 18, 21, 28, 35],
    windows: List[int] = [7, 14, 28],
    group_cols: List[str] = ["store_nbr", "family"],
    target_col: str = "sales"
) -> pd.DataFrame:
    """
    Generates strict anti-leakage lag and rolling features.
    
    CRITICAL ANTI-LEAKAGE RULE:
    The forecast horizon is exactly 16 days (2017-08-16 to 2017-08-31).
    All lags MUST be >= 16 days so they are 100% available at prediction time.
    All rolling statistics are computed on the shift(16) base series.
    """
    df = df.copy()
    # Sort strictly chronologically within each time series group
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    
    grouped = df.groupby(group_cols)[target_col]
    
    # 1. Pure Lags (minimum lag = 16)
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = grouped.shift(lag).astype(np.float32)
        
    # 2. Shifted base for rolling windows (shift 16 days)
    base_shift_16 = grouped.shift(16)
    
    # 3. Rolling Statistics over shifted window
    for w in windows:
        # Rolling mean
        df[f"rolling_mean_{w}"] = (
            df.groupby(group_cols)[f"lag_16"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
            .astype(np.float32)
        )
        # Rolling standard deviation (volatility)
        df[f"rolling_std_{w}"] = (
            df.groupby(group_cols)[f"lag_16"]
            .transform(lambda x: x.rolling(w, min_periods=1).std())
            .fillna(0.0)
            .astype(np.float32)
        )
        
    # 4. Promotion Lags (onpromotion is known in test, but past promo momentum is informative)
    if "onpromotion" in df.columns:
        promo_grouped = df.groupby(group_cols)["onpromotion"]
        df["promo_lag_16"] = promo_grouped.shift(16).fillna(0).astype(np.int16)
        df["promo_rolling_mean_7"] = (
            df.groupby(group_cols)["promo_lag_16"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
            .astype(np.float32)
        )
        
    return df


def create_oil_momentum_features(df: pd.DataFrame, oil_col: str = "dcoilwtico") -> pd.DataFrame:
    """
    Computes crude oil price momentum and moving averages across dates.
    """
    df = df.copy()
    if oil_col not in df.columns:
        return df
        
    # Extract unique daily oil series to compute rolling metrics once per date
    oil_series = df[["date", oil_col]].drop_duplicates().sort_values("date").reset_index(drop=True)
    
    oil_series["oil_lag_16"] = oil_series[oil_col].shift(16).bfill().astype(np.float32)
    oil_series["oil_rolling_mean_7"] = oil_series[oil_col].rolling(7, min_periods=1).mean().astype(np.float32)
    oil_series["oil_pct_change_7"] = oil_series[oil_col].pct_change(7).fillna(0.0).astype(np.float32)
    
    # Merge back
    features_to_join = ["date", "oil_lag_16", "oil_rolling_mean_7", "oil_pct_change_7"]
    df = pd.merge(df, oil_series[features_to_join], on="date", how="left")
    return df


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepares categorical features for both Tree algorithms (LightGBM/XGBoost)
    and Linear models (Ridge):
    - Converts categories to pandas 'category' type for native LightGBM handling.
    - Creates integer label encodings for XGBoost.
    """
    df = df.copy()
    cat_cols = ["store_nbr", "family", "city", "state", "type", "cluster"]
    
    for col in cat_cols:
        if col in df.columns:
            df[f"{col}_cat"] = df[col].astype("category")
            df[f"{col}_code"] = df[col].astype("category").cat.codes.astype(np.int16)
            
    return df, cat_cols


def build_full_feature_pipeline(
    sales_df: pd.DataFrame,
    clean_hols_df: pd.DataFrame
) -> pd.DataFrame:
    """
    End-to-end feature pipeline orchestrator.
    Executes all transformations in sequence, producing the final modeling dataframe.
    """
    print("[FeatureEngineering] 1. Generating calendar & trigonometric features...")
    df = create_calendar_features(sales_df)
    
    print("[FeatureEngineering] 2. Computing holiday proximity indicators...")
    df = create_holiday_proximity_features(df, clean_hols_df)
    
    print("[FeatureEngineering] 3. Computing oil momentum indicators...")
    df = create_oil_momentum_features(df)
    
    print("[FeatureEngineering] 4. Building anti-leakage lag & rolling statistics (lags >= 16)...")
    df = create_lag_and_rolling_features(df)
    
    print("[FeatureEngineering] 5. Encoding categorical hierarchies...")
    df, _ = encode_categorical_features(df)
    
    print(f"[FeatureEngineering] Pipeline complete! Resulting shape: {df.shape}")
    return df
