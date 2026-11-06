import pytest
from src.eval import extract_numerical_claims, verify_claims_against_evidence, audit_causal_assumptions
from src.assistant import (
    answer_with_evidence,
    build_research_context,
    MODE_GROUNDED,
    MODE_UNCONSTRAINED,
    MODE_AUDITOR,
    KNOWLEDGE_SOURCE_DATASET,
    KNOWLEDGE_SOURCE_DOCUMENTS,
    KNOWLEDGE_SOURCE_HYBRID,
)
import pandas as pd


def test_extract_numerical_claims():
    text = "The estimated ATE is 2.45 with a 95% CI of [1.80, 3.10], p = 0.002, and 45.2% bias reduction. Year 2024 was excluded."
    nums = extract_numerical_claims(text)
    assert 2.45 in nums
    assert 1.80 in nums
    assert 3.10 in nums
    assert 0.002 in nums
    assert 45.2 in nums
    assert 0.452 in nums  # percentage converted
    # 2024 is year boilerplate and should be excluded
    assert 2024.0 not in nums


def test_verify_claims_against_evidence():
    context = {
        "causal_result": {
            "ate": 2.45,
            "ci_low": 1.80,
            "ci_high": 3.10,
            "p_value": 0.002,
            "n": 600
        }
    }

    # Case 1: Grounded response
    grounded_text = "The adjusted ATE is 2.45 (95% CI [1.80, 3.10], p=0.002) across n=600 observations."
    eval_res = verify_claims_against_evidence(grounded_text, context)
    assert eval_res["faithfulness_score"] >= 90.0
    assert eval_res["unsupported_claims"] == 0

    # Case 2: Hallucinated response
    hallucinated_text = "The treatment effect is 99.8 with 8765 patients."
    eval_hallucinated = verify_claims_against_evidence(hallucinated_text, context)
    assert eval_hallucinated["unsupported_claims"] >= 1
    assert eval_hallucinated["faithfulness_score"] < 50.0


def test_audit_causal_assumptions():
    # Rigorous text disclosing assumptions
    rigorous_text = """
    Under the assumption of unconfoundedness (no unmeasured confounding) and positivity (overlap),
    along with SUTVA (no interference between units) and consistency, the estimated effect is causal.
    """
    audit = audit_causal_assumptions(rigorous_text)
    assert audit["disclosed_count"] >= 3
    assert audit["has_overclaiming"] is False
    assert "Exemplary" in audit["rigor_rating"] or "Acceptable" in audit["rigor_rating"]

    # Overclaiming text
    overclaiming_text = "This analysis definitively proves that the treatment caused the outcome with direct proof."
    audit_overclaim = audit_causal_assumptions(overclaiming_text)
    assert audit_overclaim["has_overclaiming"] is True


def test_assistant_without_key():
    # Should not raise exception, but return informative message and dict structure
    res = answer_with_evidence("What is the effect?", {}, api_key=None)
    assert isinstance(res, dict)
    assert "text" in res
    assert "faithfulness" in res
    assert "assumptions_audit" in res
    assert res["error"] == "MISSING_API_KEY"


def test_build_research_context():
    df = pd.DataFrame({"t": [0, 1] * 20, "y": range(40), "x1": range(40)})
    ctx = build_research_context(
        df=df,
        dataset_name="demo",
        analysis_spec={"treatment": "t", "outcome": "y", "confounders": ["x1"]},
        all_estimators=[{"estimator": "OLS", "ate": 1.0, "se": 0.1, "ci_low": 0.8, "ci_high": 1.2}],
        sensitivity_result={"e_value_point": 2.5, "e_value_ci": 1.8, "robustness_rating": "Moderate"}
    )
    assert "dataset_profile" in ctx
    assert ctx["dataset_profile"]["total_observations"] == 40
    assert "causal_specification" in ctx
    assert "causal_estimators_evaluated" in ctx
    assert "sensitivity_analysis_e_value" in ctx
    assert "missing_evidence_and_unobserved_limits" in ctx


def test_assistant_dual_knowledge_sources():
    # 1. Dataset source offline answer
    df = pd.DataFrame({"treatment": [0, 1] * 25, "outcome": [1.2, 3.5] * 25, "covar": range(50)})
    ctx_dataset = build_research_context(
        df=df,
        dataset_name="clinical_cohort.csv",
        analysis_spec={"treatment": "treatment", "outcome": "outcome", "confounders": ["covar"]},
        all_estimators=[
            {"estimator": "Doubly Robust AIPW", "ate": 2.30, "se": 0.15, "ci_low": 2.00, "ci_high": 2.60, "p_value": 0.001},
            {"estimator": "Double Machine Learning", "ate": 2.28, "se": 0.16, "ci_low": 1.97, "ci_high": 2.59, "p_value": 0.001}
        ],
        sensitivity_result={"e_value_point": 2.85, "e_value_ci": 1.95, "robustness_rating": "Robust"},
        knowledge_source=KNOWLEDGE_SOURCE_DATASET
    )
    ans_ds = answer_with_evidence(
        question="What is the estimated treatment effect and sensitivity E-value?",
        context=ctx_dataset,
        api_key=None,
        knowledge_source=KNOWLEDGE_SOURCE_DATASET
    )
    assert "Doubly Robust AIPW" in ans_ds["text"]
    assert "2.3" in ans_ds["text"] or "2.30" in ans_ds["text"]
    assert "2.85" in ans_ds["text"]
    assert ans_ds["faithfulness"]["faithfulness_score"] >= 80.0

    # 2. Document source with mock vector store
    class MockVectorStore:
        def search_documents(self, query, top_k=5, paper_filter=None, file_type_filter=None, api_key=None):
            return [
                {
                    "title": "Double ML for Treatment Effects (Chernozhukov 2018)",
                    "file_name": "chernozhukov_2018.pdf",
                    "file_type": "pdf",
                    "page_num": 12,
                    "section": "Methodology",
                    "similarity_score": 0.94,
                    "text": "Neyman orthogonality eliminates first-order bias from high-dimensional nuisance parameter estimation."
                }
            ]

    mock_store = MockVectorStore()
    ctx_docs = build_research_context(knowledge_source=KNOWLEDGE_SOURCE_DOCUMENTS)
    ans_docs = answer_with_evidence(
        question="How does Double ML achieve debiasing?",
        context=ctx_docs,
        api_key=None,
        knowledge_source=KNOWLEDGE_SOURCE_DOCUMENTS,
        vector_store=mock_store
    )
    assert len(ans_docs["citations"]) == 1
    assert "Chernozhukov" in ans_docs["text"]
    assert ans_docs["citations"][0]["page"] == 12
    assert ans_docs["citations"][0]["file_type"] == "pdf"

    # 3. Hybrid source combining dataset and document store
    ans_hybrid = answer_with_evidence(
        question="Compare my empirical AIPW estimate with Double ML theory.",
        context=ctx_dataset,
        api_key=None,
        knowledge_source=KNOWLEDGE_SOURCE_HYBRID,
        vector_store=mock_store
    )
    assert "Empirical Dataset Results" in ans_hybrid["text"]
    assert "Retrieved Document Passages" in ans_hybrid["text"] or "Chernozhukov" in ans_hybrid["text"]
    assert len(ans_hybrid["citations"]) >= 1


