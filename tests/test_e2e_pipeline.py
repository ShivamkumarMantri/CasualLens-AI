import json
import numpy as np
import pandas as pd
import pytest

from src.profiler import profile_dataframe, correlation_table, detect_outliers, compute_baseline_smd
from src.dag import CausalGraph
from src.causal import compare_all_estimators, estimate_ate
from src.sensitivity import compute_e_value, run_placebo_treatment_refutation, run_random_common_cause_refutation, run_data_subset_refutation
from src.benchmark import generate_synthetic_dataset, run_monte_carlo_benchmark
from src.assistant import answer_with_evidence, MODE_GROUNDED, MODE_AUDITOR
from src.eval import verify_claims_against_evidence, audit_causal_assumptions
from src.reporting import build_markdown_report, build_json_reproducibility_bundle, build_html_report


def test_full_research_pipeline_e2e():
    """
    Comprehensive End-to-End Integration Test verifying the entire research pipeline.
    """
    # 1. Dataset Generation
    df, meta = generate_synthetic_dataset(
        n=400,
        true_ate=2.5,
        confounding_strength=1.0,
        non_linear=False,
        seed=101
    )
    assert len(df) == 400
    treatment = "treatment"
    outcome = "outcome"
    covariates = meta["observed_covariates"]

    # 2. Data Profiling & Outlier Detection
    prof = profile_dataframe(df)
    assert len(prof) == len(df.columns)
    corr = correlation_table(df)
    assert len(corr) >= 5
    outliers = detect_outliers(df)
    assert len(outliers) >= 1

    # 3. DAG Backdoor Identification & Acyclicity Validation
    dag = CausalGraph(
        treatment=treatment,
        outcome=outcome,
        confounders=covariates,
        mediators=[],
        colliders=[]
    )
    dag_audit = dag.validate_variable_roles(covariates)
    assert dag_audit["is_identified"] is True
    assert dag.is_acyclic() is True
    
    # Backdoor adjustment sets discovery
    adj_discovery = dag.find_valid_adjustment_sets()
    assert adj_discovery["is_identified"] is True
    assert len(adj_discovery["minimal_sets"]) >= 1

    # Strict cycle prevention check
    is_valid_cycle, cycle_err, _ = dag.validate_edge(outcome, treatment)
    assert is_valid_cycle is False
    assert "Acyclicity Violation" in cycle_err

    mermaid_spec = dag.generate_mermaid_spec(covariates)
    assert "graph LR" in mermaid_spec

    # 4. Covariate Balance & Overlap
    smd_df = compute_baseline_smd(df, treatment, covariates)
    assert len(smd_df) == len(covariates)

    # 5. Multi-Estimator Causal Workbench
    estimators = compare_all_estimators(df, treatment, outcome, covariates)
    assert len(estimators) == 5
    ols_res = next(r for r in estimators if "OLS" in r["estimator"])
    matching_res = next(r for r in estimators if "Matching" in r["estimator"])
    ipw_res = next(r for r in estimators if "IPW" in r["estimator"])
    aipw_res = next(r for r in estimators if "AIPW" in r["estimator"])
    dml_res = next(r for r in estimators if "DML" in r["estimator"])

    for r in [ols_res, matching_res, ipw_res, aipw_res, dml_res]:
        assert not np.isnan(r["ate"])
        assert 1.5 <= r["ate"] <= 3.5  # true is 2.5
        assert r["ci_low"] <= r["ate"] <= r["ci_high"]

    # 6. Sensitivity Analysis & E-Value
    e_val = compute_e_value(
        ate=ols_res["ate"],
        se=ols_res["se"],
        outcome_sd=float(df[outcome].std()),
        ci_low=ols_res["ci_low"],
        ci_high=ols_res["ci_high"]
    )
    assert e_val["e_value_point"] > 1.2
    assert e_val["e_value_ci"] >= 1.0

    # 7. Refutation Battery
    ref_placebo = run_placebo_treatment_refutation(df, treatment, outcome, covariates, n_simulations=5)
    ref_noise = run_random_common_cause_refutation(df, treatment, outcome, covariates)
    ref_sub = run_data_subset_refutation(df, treatment, outcome, covariates, n_iterations=4)

    assert ref_placebo["passed"] is True
    assert ref_noise["passed"] is True
    assert ref_sub["passed"] is True

    # 8. Monte Carlo Simulation Lab
    mc_res = run_monte_carlo_benchmark(
        n_replications=3,
        n_samples=200,
        true_ate=2.5,
        estimators=["OLS", "AIPW"],
        seed_start=300
    )
    assert len(mc_res["summary_table"]) == 2

    # 9. Structured Context & AI Assistant Verification
    context = {
        "dataset": "synthetic_test",
        "n": len(df),
        "causal_estimators": estimators,
        "sensitivity": e_val,
        "refutations": [ref_placebo, ref_noise, ref_sub]
    }

    # Verify claim evaluation
    simulated_assistant_response = (
        f"The estimated ATE via OLS is {ols_res['ate']:.4f} with a 95% CI of [{ols_res['ci_low']:.4f}, {ols_res['ci_high']:.4f}]. "
        f"Under unconfoundedness and positivity, the effect is statistically significant."
    )
    faithfulness = verify_claims_against_evidence(simulated_assistant_response, context)
    assert faithfulness["faithfulness_score"] >= 90.0

    assumption_audit = audit_causal_assumptions(simulated_assistant_response)
    assert assumption_audit["disclosed_count"] >= 2

    # 10. Publication-Grade Reporting & Export
    md_report = build_markdown_report(
        df=df,
        causal_result=ols_res,
        analysis_spec={"treatment": treatment, "outcome": outcome, "confounders": covariates},
        all_estimators=estimators,
        balance_table=matching_res.get("balance_table"),
        sensitivity_result=e_val,
        refutation_results=[ref_placebo, ref_noise, ref_sub],
        dag_summary=dag_audit,
        assistant_logs=[{
            "question": "What is the ATE?",
            "response": simulated_assistant_response,
            "faithfulness": faithfulness,
            "assumptions_audit": assumption_audit
        }]
    )
    assert "# CausalLens AI" in md_report
    assert "## 4. Causal Estimators and Benchmarking" in md_report
    assert "## 6. Sensitivity Analysis (E-Value)" in md_report

    json_bundle = build_json_reproducibility_bundle(
        df=df,
        analysis_spec={"treatment": treatment, "outcome": outcome, "confounders": covariates},
        all_estimators=estimators,
        balance_table=matching_res.get("balance_table"),
        sensitivity_result=e_val,
        refutation_results=[ref_placebo, ref_noise, ref_sub]
    )
    parsed_json = json.loads(json_bundle)
    assert parsed_json["metadata"]["n_observations"] == 400

    html_report = build_html_report(md_report)
    assert "<!DOCTYPE html>" in html_report
    assert "CausalLens AI" in html_report


def test_app_compiles_and_parses_ast():
    """Verify that app.py compiles cleanly to bytecode without syntax or indentation errors."""
    import ast
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code, filename="app.py", mode="exec")
    assert tree is not None
    compiled = compile(code, "app.py", "exec")
    assert compiled is not None


def test_app_navigation_constants_and_all_pages():
    """Verify that navigation constants and ALL_PAGES are defined and valid."""
    import ast
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    # Check that ALL_PAGES is assigned
    assert "ALL_PAGES" in code
    assert "all_pages" in code
    assert "PAGE_OVERVIEW" in code
    assert "PAGE_SETTINGS" in code



