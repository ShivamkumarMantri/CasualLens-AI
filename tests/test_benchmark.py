import numpy as np
import pandas as pd
from src.benchmark import generate_synthetic_dataset, run_monte_carlo_benchmark


def test_generate_synthetic_dataset():
    df, meta = generate_synthetic_dataset(n=300, true_ate=1.5, confounding_strength=1.0, seed=42)
    assert len(df) == 300
    assert "treatment" in df.columns
    assert "outcome" in df.columns
    assert meta["true_ate"] == 1.5
    assert set(df["treatment"].unique()) == {0, 1}


def test_generate_non_linear_and_unmeasured():
    df, meta = generate_synthetic_dataset(
        n=200, true_ate=2.0, non_linear=True, unmeasured_confounding=True, seed=99
    )
    assert len(df) == 200
    assert "_latent_unmeasured_u" in df.columns
    assert meta["non_linear"] is True
    assert meta["unmeasured_confounding"] is True


def test_monte_carlo_benchmark_quick():
    # Quick 3-replication test to verify execution, metrics computation, and coverage
    res = run_monte_carlo_benchmark(
        n_replications=3,
        n_samples=250,
        true_ate=2.0,
        estimators=["OLS", "AIPW"],
        seed_start=123
    )
    assert "summary_table" in res
    st = res["summary_table"]
    assert len(st) == 2
    assert "bias" in st.columns
    assert "rmse" in st.columns
    assert "coverage_95_pct" in st.columns

