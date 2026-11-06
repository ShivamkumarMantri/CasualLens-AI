import numpy as np
import pandas as pd
import pytest
from src.causal import (
    estimate_ate,
    propensity_score_match,
    estimate_ipw,
    estimate_aipw,
    estimate_dml,
    compare_all_estimators
)


@pytest.fixture
def synthetic_observational_data():
    """
    Synthetic dataset with true treatment effect tau = 2.5.
    Confounding from age and prior_visits.
    """
    rng = np.random.default_rng(42)
    n = 600
    age = rng.normal(40, 10, n).clip(18, 70)
    prior = rng.poisson(2, n)
    # Propensity model
    logit_p = -1.2 + 0.03 * age + 0.25 * prior
    prob = 1 / (1 + np.exp(-logit_p))
    t = rng.binomial(1, prob)
    # True causal effect is 2.5
    y = 2.5 * t + 0.05 * age - 0.4 * prior + rng.normal(0, 1.5, n)
    return pd.DataFrame({"age": age, "prior": prior, "treat": t, "outcome": y})


def test_ols_estimator(synthetic_observational_data):
    df = synthetic_observational_data
    res = estimate_ate(df, "treat", "outcome", ["age", "prior"])
    assert "ate" in res
    assert "att" in res
    assert 2.0 <= res["ate"] <= 3.0  # true is 2.5
    assert 2.0 <= res["att"] <= 3.0
    assert res["ci_low"] < res["ate"] < res["ci_high"]
    assert res["att_ci_low"] <= res["att"] <= res["att_ci_high"]
    assert res["se"] > 0
    assert res["p_value"] < 0.05
    assert len(res["coefficients"]) == 4  # const + treat + age + prior
    assert "identification_assumptions" in res
    assert "unconfoundedness" in res["identification_assumptions"]


def test_psm_estimator(synthetic_observational_data):
    df = synthetic_observational_data
    res = propensity_score_match(df, "treat", "outcome", ["age", "prior"], caliper_sd=0.25)
    assert "ate" in res
    assert "att" in res
    assert 1.8 <= res["att"] <= 3.2
    assert 1.8 <= res["ate"] <= 3.2
    assert res["att_ci_low"] <= res["att"] <= res["att_ci_high"]
    assert res["n_pairs"] >= 30
    assert len(res["balance_table"]) == 2
    assert "overlap_diagnostics" in res
    assert res["overlap_diagnostics"]["pct_treated_on_support"] >= 80.0
    # Verify balance improvement
    for b in res["balance_table"]:
        assert abs(b["smd_post"]) <= 0.25


def test_ipw_estimator(synthetic_observational_data):
    df = synthetic_observational_data
    res = estimate_ipw(df, "treat", "outcome", ["age", "prior"])
    assert "ate" in res
    assert "att" in res
    assert 1.8 <= res["ate"] <= 3.2
    assert 1.8 <= res["att"] <= 3.2
    assert res["ci_low"] < res["ate"] < res["ci_high"]
    assert res["att_ci_low"] < res["att"] < res["att_ci_high"]
    assert res["effective_sample_size"]["total_ess"] > 100
    assert len(res["balance_table"]) == 2
    assert "overlap_diagnostics" in res


def test_aipw_estimator(synthetic_observational_data):
    df = synthetic_observational_data
    res = estimate_aipw(df, "treat", "outcome", ["age", "prior"])
    assert "ate" in res
    assert "att" in res
    assert 1.9 <= res["ate"] <= 3.1
    assert 1.9 <= res["att"] <= 3.1
    assert res["ci_low"] < res["ate"] < res["ci_high"]
    assert res["att_ci_low"] < res["att"] < res["att_ci_high"]
    assert res["se"] > 0
    assert res["p_value"] < 0.05
    assert "identification_assumptions" in res


def test_dml_estimator(synthetic_observational_data):
    df = synthetic_observational_data
    res = estimate_dml(df, "treat", "outcome", ["age", "prior"], n_splits=5)
    assert "ate" in res
    assert "att" in res
    assert 1.8 <= res["ate"] <= 3.2
    assert 1.8 <= res["att"] <= 3.2
    assert res["ci_low"] < res["ate"] < res["ci_high"]
    assert res["se"] > 0
    assert "overlap_diagnostics" in res


def test_compare_all_estimators(synthetic_observational_data):
    df = synthetic_observational_data
    all_res = compare_all_estimators(df, "treat", "outcome", ["age", "prior"])
    assert len(all_res) == 5
    for r in all_res:
        assert "estimator" in r
        assert "ate" in r
        assert "att" in r
        assert not np.isnan(r["ate"])
        assert not np.isnan(r["att"])
        assert "identification_assumptions" in r


def test_treatment_encoding_variants():
    # Test boolean treatment
    df_bool = pd.DataFrame({
        "t": [True, False] * 30,
        "x": list(range(60)),
        "y": [1.5 if i % 2 == 0 else 0.0 for i in range(60)]
    })
    res = estimate_ate(df_bool, "t", "y", ["x"])
    assert "ate" in res
    assert "att" in res

    # Test string treatment
    df_str = pd.DataFrame({
        "t": ["Treated", "Control"] * 30,
        "x": list(range(60)),
        "y": [2.0 if i % 2 == 0 else 0.0 for i in range(60)]
    })
    res_str = estimate_ate(df_str, "t", "y", ["x"])
    assert "ate" in res_str
    assert "att" in res_str


def test_causal_assumptions_identification_separation(synthetic_observational_data):
    df = synthetic_observational_data
    estimators = compare_all_estimators(df, "treat", "outcome", ["age", "prior"])
    for r in estimators:
        # Every estimator must document theoretical assumptions
        assert "identification_assumptions" in r
        assumptions = r["identification_assumptions"]
        assert isinstance(assumptions, dict)
        assert "unconfoundedness" in assumptions
        assert "positivity" in assumptions
        assert "sutva" in assumptions
        
        # Every estimator must provide empirically verified statistics
        assert "ate" in r and not np.isnan(r["ate"])
        assert "att" in r and not np.isnan(r["att"])
        assert "se" in r and r["se"] > 0
        assert "ci_low" in r and "ci_high" in r
        assert r["ci_low"] <= r["ci_high"]


def test_excel_and_csv_dataset_ingestion(tmp_path):
    # Test writing and reading both CSV and Excel
    rng = np.random.default_rng(99)
    df_test = pd.DataFrame({
        "feature_1": rng.normal(0, 1, 50),
        "feature_2": rng.normal(5, 2, 50),
        "treatment": rng.binomial(1, 0.5, 50),
        "outcome": rng.normal(10, 3, 50)
    })
    
    csv_file = tmp_path / "test_data.csv"
    xlsx_file = tmp_path / "test_data.xlsx"
    
    df_test.to_csv(csv_file, index=False)
    df_test.to_excel(xlsx_file, index=False)
    
    df_csv_loaded = pd.read_csv(csv_file)
    df_xlsx_loaded = pd.read_excel(xlsx_file)
    
    assert len(df_csv_loaded) == 50
    assert len(df_xlsx_loaded) == 50
    
    res_csv = estimate_ate(df_csv_loaded, "treatment", "outcome", ["feature_1", "feature_2"])
    res_xlsx = estimate_ate(df_xlsx_loaded, "treatment", "outcome", ["feature_1", "feature_2"])
    
    assert np.isclose(res_csv["ate"], res_xlsx["ate"], atol=1e-5)
    assert np.isclose(res_csv["se"], res_xlsx["se"], atol=1e-5)

