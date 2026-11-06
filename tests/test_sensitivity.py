import numpy as np
import pandas as pd
import pytest
from src.sensitivity import (
    compute_e_value,
    compute_dual_e_values,
    run_placebo_treatment_refutation,
    run_random_common_cause_refutation,
    run_data_subset_refutation,
)


@pytest.fixture
def sensitivity_test_data():
    rng = np.random.default_rng(123)
    n = 500
    x = rng.normal(0, 1, n)
    t = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x)))
    y = 2.0 * t + 1.2 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "t": t, "y": y})


def test_compute_e_value():
    # Moderate effect: ATE=2.0, SD=2.0 -> d=1.0 -> RR ~ exp(0.91) ~ 2.48 -> E-value ~ 4.4
    ev = compute_e_value(ate=2.0, se=0.2, outcome_sd=2.0, ci_low=1.6, ci_high=2.4)
    assert ev["e_value_point"] > 1.5
    assert ev["e_value_ci"] > 1.0
    assert "High robustness" in ev["robustness_rating"] or "Moderate" in ev["robustness_rating"]

    # Null effect
    ev_null = compute_e_value(ate=0.05, se=0.2, outcome_sd=2.0, ci_low=-0.3, ci_high=0.4)
    assert ev_null["e_value_ci"] == 1.0


def test_placebo_treatment_refutation(sensitivity_test_data):
    df = sensitivity_test_data
    ref = run_placebo_treatment_refutation(df, "t", "y", ["x"], n_simulations=5, random_state=42)
    assert ref["test"] == "Placebo Treatment Permutation Refutation"
    assert ref["passed"] is True
    assert abs(ref["mean_placebo_ate"]) < abs(ref["original_ate"])


def test_random_common_cause_refutation(sensitivity_test_data):
    df = sensitivity_test_data
    ref = run_random_common_cause_refutation(df, "t", "y", ["x"], random_state=42)
    assert ref["passed"] is True
    assert ref["percent_change"] < 10.0


def test_data_subset_refutation(sensitivity_test_data):
    df = sensitivity_test_data
    ref = run_data_subset_refutation(df, "t", "y", ["x"], subset_fraction=0.8, n_iterations=5, random_state=42)
    assert ref["passed"] is True
    assert ref["percent_drift"] < 15.0


def test_dual_e_values():
    dual = compute_dual_e_values(ate=2.0, att=2.2, outcome_sd=2.0, ate_ci=(1.5, 2.5), att_ci=(1.6, 2.8))
    assert "ate_e_value" in dual
    assert "att_e_value" in dual
    assert dual["ate_e_value"]["e_value_point"] > 1.5
    assert dual["att_e_value"]["e_value_point"] > 1.5

