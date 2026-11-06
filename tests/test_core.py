import pandas as pd
import numpy as np
from src.profiler import profile_dataframe, correlation_table, detect_outliers, compute_baseline_smd
from src.causal import estimate_ate

def test_profile():
    df = pd.DataFrame({"x": [1, 2, None], "y": [2, 4, 6]})
    out = profile_dataframe(df)
    assert int(out.loc[out.column == "x", "missing"].iloc[0]) == 1
    assert "mean" in out.columns
    assert float(out.loc[out.column == "y", "mean"].iloc[0]) == 4.0

def test_correlation_table():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 3, 2, 1]})
    corr = correlation_table(df)
    assert len(corr) == 3
    assert abs(corr.iloc[0]["pearson_r"]) >= 0.99

def test_outlier_detection():
    df = pd.DataFrame({"normal": list(range(100)), "spiked": list(range(99)) + [10000]})
    out = detect_outliers(df)
    assert len(out) == 2
    spiked_row = out[out.feature == "spiked"].iloc[0]
    assert spiked_row["iqr_outliers"] >= 1

def test_baseline_smd():
    df = pd.DataFrame({
        "t": [0, 0, 0, 1, 1, 1],
        "x1": [10.0, 11.0, 10.5, 20.0, 21.0, 20.5],
        "x2": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    })
    smd_df = compute_baseline_smd(df, "t", ["x1", "x2"])
    assert len(smd_df) == 2
    x1_row = smd_df[smd_df.covariate == "x1"].iloc[0]
    assert x1_row["abs_smd"] > 1.0  # Large imbalance
    x2_row = smd_df[smd_df.covariate == "x2"].iloc[0]
    assert x2_row["abs_smd"] < 0.1

def test_ate_positive_signal():
    df = pd.DataFrame({"t": [0, 1] * 50, "x": list(range(100)), "y": [0, 1] * 50})
    result = estimate_ate(df, "t", "y", ["x"])
    assert "ate" in result
