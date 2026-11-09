"""
Unit test suite for Step 9 (Ensembling) and Step 10 (Submission Generation).
Validates weighted blending, variance reduction, expm1 inversion, and Kaggle submission formatting.
"""

import numpy as np
import pandas as pd
from src.ensemble import calculate_blend_predictions, evaluate_ensemble_blend
from src.submission import format_and_save_submission, verify_submission_format


def test_ensemble_blending():
    np.random.seed(42)
    N = 100
    y_true = np.random.uniform(10, 100, N)
    
    # Model 1 has slight positive bias, Model 2 has slight negative bias
    pred1 = y_true + np.random.normal(5, 2, N)
    pred2 = y_true + np.random.normal(-5, 2, N)
    
    preds_dict = {"Model_A": pred1, "Model_B": pred2}
    weights_dict = {"Model_A": 0.5, "Model_B": 0.5}
    
    rmsle_blend, blend_preds = evaluate_ensemble_blend(preds_dict, weights_dict, y_true)
    
    assert len(blend_preds) == N
    assert rmsle_blend < 0.20
    print("[PASS] test_ensemble_blending passed!")


def test_submission_generation():
    test_ids = np.arange(3000888, 3000888 + 28512)
    # Mock log predictions with some negative values to test clipping
    log_preds = np.random.normal(3.5, 0.8, len(test_ids))
    log_preds[0] = -2.0  # Should be clipped to 0
    
    sub = format_and_save_submission(
        test_ids, 
        log_preds, 
        output_path="outputs/submission.csv"
    )
    
    assert len(sub) == 28512
    assert sub.loc[0, "sales"] == 0.0, "Negative prediction must be clipped to 0.0"
    assert verify_submission_format(sub, 28512)
    print("[PASS] test_submission_generation passed!")


if __name__ == "__main__":
    test_ensemble_blending()
    test_submission_generation()
    print("\n>>> ALL STEP 9 & STEP 10 TESTS PASSED! <<<")
