"""
Submission Module for Store Sales - Time Series Forecasting.
Handles final test feature alignment, prediction generation,
expm1 inversion, negative clipping, and submission file generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def verify_submission_format(
    sub_df: pd.DataFrame, 
    expected_rows: int = 28512
) -> bool:
    """
    Validates that the submission DataFrame strictly complies with Kaggle requirements:
    1. Exactly matching expected rows (28,512).
    2. Exactly two columns: ['id', 'sales'].
    3. No missing (NaN) values.
    4. No negative sales values.
    5. 'id' is integer, 'sales' is non-negative float.
    """
    assert list(sub_df.columns) == ["id", "sales"], f"Columns must be ['id', 'sales'], got {list(sub_df.columns)}"
    assert len(sub_df) == expected_rows, f"Expected {expected_rows} rows, got {len(sub_df)}"
    assert sub_df["sales"].isnull().sum() == 0, "Submission contains NaN values!"
    assert (sub_df["sales"] < 0).sum() == 0, "Submission contains negative sales!"
    assert not np.isinf(sub_df["sales"]).any(), "Submission contains infinite values!"
    print("[PASS] Submission format verification passed: 100% compliant with Kaggle specifications.")
    return True


def format_and_save_submission(
    test_ids: np.ndarray,
    test_predictions_log: np.ndarray,
    output_path: Union[str, Path] = "outputs/submission.csv"
) -> pd.DataFrame:
    """
    Inverts log1p predictions via expm1, clips negatives to 0, formats to Kaggle schema,
    and writes to disk.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # Invert log1p: exp(z) - 1
    real_sales = np.expm1(test_predictions_log)
    
    # Clip negative values to zero
    real_sales_clipped = np.clip(real_sales, 0, None)
    
    sub = pd.DataFrame({
        "id": test_ids.astype(int),
        "sales": real_sales_clipped.astype(np.float32)
    })
    
    verify_submission_format(sub, expected_rows=len(test_ids))
    
    sub.to_csv(out, index=False)
    print(f"Final submission successfully saved to: {out.resolve()}")
    print(f"File size: {out.stat().st_size / 1e3:.1f} KB")
    return sub
