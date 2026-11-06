import json
import pandas as pd
from src.reporting import build_markdown_report, build_json_reproducibility_bundle, build_html_report


def test_markdown_report_minimal():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    report = build_markdown_report(df)
    assert "# CausalLens AI" in report
    assert "Sample Size (N)" in report
    assert "Features / Columns" in report


def test_markdown_report_with_matching_none_values():
    # Verify the bug where ci_low is None or missing does not crash
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    causal_result = {
        "method": "Matching",
        "ate": 1.25,
        "ci_low": None,
        "ci_high": None,
        "p_value": None
    }
    report = build_markdown_report(df, causal_result=causal_result)
    assert "Matching" in report
    assert "1.2500" in report
    assert "N/A" in report


def test_markdown_report_comprehensive():
    df = pd.DataFrame({"t": [0, 1] * 20, "x": range(40), "y": range(40)})
    all_estimators = [
        {"estimator": "OLS", "ate": 2.1, "se": 0.15, "ci_low": 1.8, "ci_high": 2.4, "p_value": 0.001, "n": 40},
        {"estimator": "AIPW", "ate": 2.05, "se": 0.16, "ci_low": 1.73, "ci_high": 2.37, "p_value": 0.001, "n": 40}
    ]
    sensitivity = {
        "e_value_point": 2.8,
        "e_value_ci": 1.9,
        "robustness_rating": "Moderate",
        "interpretation": "Confounder RR must exceed 2.8"
    }
    refutation = [
        {"test": "Placebo Test", "original_ate": 2.1, "mean_placebo_ate": 0.02, "passed": True, "status": "PASSED"}
    ]
    report = build_markdown_report(
        df,
        all_estimators=all_estimators,
        sensitivity_result=sensitivity,
        refutation_results=refutation
    )
    assert "## 4. Causal Estimators and Benchmarking" in report
    assert "## 6. Sensitivity Analysis (E-Value)" in report
    assert "Placebo Test" in report
    assert "E-value" in report


def test_json_reproducibility_bundle():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    bundle_str = build_json_reproducibility_bundle(
        df=df,
        analysis_spec={"treatment": "a", "outcome": "b"},
        all_estimators=[{"estimator": "OLS", "ate": 1.0}]
    )
    data = json.loads(bundle_str)
    assert data["metadata"]["n_observations"] == 2
    assert data["analysis_spec"]["treatment"] == "a"
    assert len(data["estimators"]) == 1


def test_html_report_generation():
    md = "# Title\n## Section\n- Item 1\n---"
    html = build_html_report(md)
    assert "<!DOCTYPE html>" in html
    assert "<h1" in html
    assert "<h2" in html
    assert "<li>Item 1</li>" in html


def test_markdown_report_with_literature_summary():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    lit_summary = {
        "papers_count": 2,
        "chunks_count": 12,
        "papers": [
            {"title": "Pearl Causality 2009", "file_name": "pearl2009.pdf", "total_pages": 400, "total_chunks": 10},
            {"title": "Chernozhukov DML 2018", "file_name": "dml2018.pdf", "total_pages": 68, "total_chunks": 2}
        ],
        "recent_qa": {
            "query": "What is Neyman orthogonality?",
            "answer": "Neyman orthogonality makes the score insensitive to small perturbations in nuisance estimators."
        }
    }
    report = build_markdown_report(df, literature_summary=lit_summary)
    assert "## 9. Academic Literature & Grounded Citations (RAG)" in report
    assert "Pearl Causality 2009" in report
    assert "Neyman orthogonality" in report

    bundle = build_json_reproducibility_bundle(
        df=df,
        analysis_spec={},
        all_estimators=[],
        literature_summary=lit_summary
    )
    data = json.loads(bundle)
    assert data["literature_summary"]["papers_count"] == 2
