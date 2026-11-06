import os
import pytest
from src.gemini_client import (
    get_api_key,
    mask_api_key,
    sanitize_message,
    validate_gemini_connection,
    generate_gemini_text,
    generate_gemini_embedding,
    DEFAULT_TEXT_MODELS,
)
from src.assistant import (
    answer_with_evidence,
    build_research_context,
    KNOWLEDGE_SOURCE_DATASET,
    KNOWLEDGE_SOURCE_DOCUMENTS,
    MODE_GROUNDED,
)
from src.rag import (
    AcademicVectorStore,
    answer_document_query,
    answer_document_chat,
)


def test_env_key_resolution_and_masking():
    """Verify that GEMINI_API_KEY is securely resolved and masked."""
    key = get_api_key()
    assert key is not None, "GEMINI_API_KEY should be resolved from .env"
    assert len(key) > 20

    masked = mask_api_key(key)
    assert masked.startswith(key[:6])
    assert masked.endswith(key[-4:])
    assert "..." in masked
    assert key not in masked, "Raw secret must never equal the masked representation"

    # Test sanitization
    leak_string = f"CRITICAL: Failed with key {key} on backend"
    sanitized = sanitize_message(leak_string, key)
    assert key not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_gemini_connection_live_ping():
    """Verify live API connection to Google GenAI using the authenticated key."""
    status = validate_gemini_connection()
    assert status["is_valid"] is True, f"Connection failed: {status}"
    assert status["status"] == "connected"
    assert status["model"] in DEFAULT_TEXT_MODELS
    assert status["masked_key"].startswith(get_api_key()[:6])
    assert status["sample_response"] is not None


def test_gemini_text_generation_live():
    """Verify text generation produces genuine non-empty responses from Google GenAI."""
    res = generate_gemini_text(
        prompt="Explain confounding in observational research in one concise sentence.",
        temperature=0.2,
    )
    assert res["success"] is True, f"Generation failed: {res.get('error')}"
    assert len(res["text"].strip()) > 10
    assert res["model_used"] in DEFAULT_TEXT_MODELS


def test_gemini_embedding_generation_live():
    """Verify dense embedding vector generation with Google GenAI."""
    emb = generate_gemini_embedding("Double Machine Learning for treatment effect estimation")
    assert emb is not None
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert all(isinstance(x, (float, int)) for x in emb[:10])


def test_assistant_online_gemini_synthesis():
    """Verify AI Research Assistant generates evidence-grounded response using live Gemini."""
    context = {
        "dataset_profile": {"name": "synthetic_benchmark", "total_observations": 500, "feature_count": 5},
        "causal_specification": {"treatment_variable": "treatment", "outcome_variable": "outcome", "adjusted_confounders": ["age", "income"]},
        "causal_estimators_evaluated": [
            {"estimator": "Double Machine Learning", "ate_point_estimate": 2.51, "ate_95_ci": [2.20, 2.82], "ate_std_error": 0.16, "ate_p_value": 0.001}
        ],
        "sensitivity_analysis_e_value": {"e_value_point_estimate": 2.85, "e_value_ci_bound": 1.95, "robustness_rating": "Robust"}
    }
    ans = answer_with_evidence(
        question="What is the estimated treatment effect and how robust is it to unmeasured confounding?",
        context=context,
        knowledge_source=KNOWLEDGE_SOURCE_DATASET,
        mode=MODE_GROUNDED,
    )
    assert ans["error"] is None
    assert ans["model"] in DEFAULT_TEXT_MODELS
    assert len(ans["text"]) > 20
    assert "Double Machine Learning" in ans["text"] or "2.5" in ans["text"] or "robust" in ans["text"].lower()


def test_rag_document_qa_online_gemini_synthesis(tmp_path):
    """Verify Document QA generates live cited answers with uploaded document passages."""
    import io
    store = AcademicVectorStore(persist_directory=str(tmp_path / "test_chroma"))
    
    file_bytes = (
        "The Intensive Hypertension Intervention Trial evaluated 1,200 cardiovascular patients. "
        "The estimated average treatment effect on systolic blood pressure reduction was 12.4 mmHg "
        "(95% CI: 10.1 to 14.7, p < 0.001) using Doubly Robust AIPW estimators."
    ).encode("utf-8")
    
    doc_file = io.BytesIO(file_bytes)
    doc_file.name = "clinical_policy_trial.txt"
    
    store.add_documents([doc_file])
    assert store.collection.count() >= 1

    ans = answer_document_query(
        query="What was the average treatment effect on systolic blood pressure reduction?",
        vector_store=store,
        top_k=2
    )
    assert ans["insufficient_evidence"] is False
    assert len(ans["citations"]) >= 1
    assert "12.4" in ans["answer"] or "blood pressure" in ans["answer"].lower()


def test_invalid_api_key_handling_never_crashes_and_sanitizes():
    """Verify that an invalid API key is caught gracefully without leaking or crashing."""
    dummy_key = "AQ.INVALID_DUMMY_KEY_FOR_TESTING_1234567890"
    status = validate_gemini_connection(api_key=dummy_key)
    assert status["is_valid"] is False
    assert status["status"] in ["invalid_key", "error"]
    assert dummy_key not in status["message"]
