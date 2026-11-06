import json
import os
import re
from typing import Dict, Any, List, Optional
import pandas as pd
from src.eval import verify_claims_against_evidence, audit_causal_assumptions
from src.gemini_client import get_api_key, generate_gemini_text, mask_api_key, sanitize_message

_ENV_DEFAULT = object()

# Prompting modes for comparative research evaluation (Ablation Study)
MODE_GROUNDED = "grounded"
MODE_UNCONSTRAINED = "unconstrained"
MODE_AUDITOR = "auditor"

# Knowledge sources supported by the AI Research Assistant
KNOWLEDGE_SOURCE_DATASET = "dataset"
KNOWLEDGE_SOURCE_DOCUMENTS = "documents"
KNOWLEDGE_SOURCE_HYBRID = "hybrid"

ALL_KNOWLEDGE_SOURCES = [
    KNOWLEDGE_SOURCE_DATASET,
    KNOWLEDGE_SOURCE_DOCUMENTS,
    KNOWLEDGE_SOURCE_HYBRID,
]


def build_research_context(
    df: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    analysis_spec: Optional[Dict[str, Any]] = None,
    all_estimators: Optional[List[Dict[str, Any]]] = None,
    baseline_smd: Optional[pd.DataFrame] = None,
    balance_table: Optional[List[Dict[str, Any]]] = None,
    sensitivity_result: Optional[Dict[str, Any]] = None,
    refutation_results: Optional[List[Dict[str, Any]]] = None,
    dag_summary: Optional[Dict[str, Any]] = None,
    retrieved_passages: Optional[List[Dict[str, Any]]] = None,
    knowledge_source: str = KNOWLEDGE_SOURCE_DATASET,
    document_catalog: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Construct a structured, evidence-grounded research context payload encompassing:
    1. Dataset provenance, distributions, and quality profile
    2. DAG identification & sufficient adjustment sets
    3. Multi-estimator causal estimates (ATE & ATT) across OLS, Matching, IPW, AIPW, DML
    4. Baseline & post-adjustment covariate balance tables (SMD)
    5. Sensitivity analysis E-values & bounds for unmeasured confounding
    6. Refutation test battery results (placebo treatment, random common cause, subset stability)
    7. Multi-format retrieved document evidence & literature catalog (when querying documents or hybrid)
    """
    context: Dict[str, Any] = {
        "active_knowledge_source": knowledge_source
    }

    # 1. Dataset Overview & Data Quality Profile (when dataset or hybrid is active)
    if df is not None and knowledge_source in [KNOWLEDGE_SOURCE_DATASET, KNOWLEDGE_SOURCE_HYBRID]:
        num_cols = list(df.select_dtypes(include="number").columns)
        cat_cols = [c for c in df.columns if c not in num_cols]
        missing_total = int(df.isna().sum().sum())
        total_cells = int(df.size)
        
        # Top pairwise correlations
        top_corrs = []
        if len(num_cols) >= 2:
            c_mat = df[num_cols].corr()
            for i, c1 in enumerate(c_mat.columns):
                for c2 in c_mat.columns[i + 1:]:
                    val = c_mat.loc[c1, c2]
                    if not pd.isna(val):
                        top_corrs.append({"pair": f"{c1} - {c2}", "pearson_r": round(float(val), 3)})
            top_corrs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

        context["dataset_profile"] = {
            "name": dataset_name or "Observational Dataset",
            "total_observations": int(len(df)),
            "feature_count": int(len(df.columns)),
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "total_missing_cells": missing_total,
            "missingness_percentage": round(missing_total / total_cells * 100, 2) if total_cells > 0 else 0.0,
            "duplicate_rows": int(df.duplicated().sum()),
            "top_correlations": top_corrs[:8]
        }

    # 2. Causal Identification Specification & SCM / DAG
    if analysis_spec and knowledge_source in [KNOWLEDGE_SOURCE_DATASET, KNOWLEDGE_SOURCE_HYBRID]:
        context["causal_specification"] = {
            "treatment_variable": analysis_spec.get("treatment"),
            "outcome_variable": analysis_spec.get("outcome"),
            "adjusted_confounders": analysis_spec.get("confounders", []),
        }

    if dag_summary and knowledge_source in [KNOWLEDGE_SOURCE_DATASET, KNOWLEDGE_SOURCE_HYBRID]:
        context["dag_identification_status"] = {
            "is_identified": dag_summary.get("is_identified", False),
            "status_text": dag_summary.get("identification_status"),
            "sufficient_adjustment_set": dag_summary.get("sufficient_adjustment_set", []),
            "missing_confounders": dag_summary.get("missing_confounders", []),
            "conditioned_mediators": dag_summary.get("conditioned_mediators", []),
            "conditioned_colliders": dag_summary.get("conditioned_colliders", []),
            "backdoor_warnings": dag_summary.get("warnings", [])
        }

    # 3. Baseline Covariate Imbalance (Unadjusted)
    if baseline_smd is not None and not baseline_smd.empty and knowledge_source in [KNOWLEDGE_SOURCE_DATASET, KNOWLEDGE_SOURCE_HYBRID]:
        context["baseline_covariate_imbalance"] = baseline_smd.to_dict(orient="records")
        max_smd = float(baseline_smd["abs_smd"].max())
        severe = baseline_smd[baseline_smd["abs_smd"] >= 0.25]["covariate"].tolist()
        context["baseline_imbalance_summary"] = {
            "maximum_baseline_smd": round(max_smd, 3),
            "severe_imbalance_covariates": severe,
            "rubin_criterion_met_baseline": max_smd < 0.10
        }

    # 4. Multi-Estimator Causal Results (ATE & ATT)
    if all_estimators and knowledge_source in [KNOWLEDGE_SOURCE_DATASET, KNOWLEDGE_SOURCE_HYBRID]:
        est_summaries = []
        for r in all_estimators:
            summary = {
                "estimator": r.get("estimator", "Unknown"),
                "method": r.get("method"),
                "ate_point_estimate": round(r["ate"], 4) if "ate" in r and r["ate"] is not None and not pd.isna(r["ate"]) else None,
                "ate_std_error": round(r["se"], 4) if "se" in r and r["se"] is not None else None,
                "ate_95_ci": [round(r["ci_low"], 4), round(r["ci_high"], 4)] if "ci_low" in r and r["ci_low"] is not None else None,
                "ate_p_value": r.get("p_value"),
                "att_point_estimate": round(r["att"], 4) if "att" in r and r["att"] is not None and not pd.isna(r["att"]) else None,
                "att_std_error": round(r.get("att_se", 0.0), 4) if "att_se" in r and r["att_se"] is not None else None,
                "att_95_ci": [round(r["att_ci_low"], 4), round(r["att_ci_high"], 4)] if "att_ci_low" in r and r["att_ci_low"] is not None else None,
                "att_p_value": r.get("att_p_value"),
                "sample_size": r.get("n", r.get("n_pairs")),
            }
            if "overlap_diagnostics" in r:
                summary["overlap_diagnostics"] = r["overlap_diagnostics"]
            est_summaries.append(summary)
        context["causal_estimators_evaluated"] = est_summaries

    # 5. Matching Post-Adjustment Covariate Balance
    if balance_table and knowledge_source in [KNOWLEDGE_SOURCE_DATASET, KNOWLEDGE_SOURCE_HYBRID]:
        context["post_matching_balance_table"] = balance_table

    # 6. Sensitivity Analysis (E-Values)
    if sensitivity_result and knowledge_source in [KNOWLEDGE_SOURCE_DATASET, KNOWLEDGE_SOURCE_HYBRID]:
        context["sensitivity_analysis_e_value"] = {
            "e_value_point_estimate": sensitivity_result.get("e_value_point"),
            "e_value_ci_bound": sensitivity_result.get("e_value_ci"),
            "standardized_effect_cohen_d": sensitivity_result.get("standardized_effect_d"),
            "approximate_risk_ratio": sensitivity_result.get("approx_risk_ratio"),
            "robustness_rating": sensitivity_result.get("robustness_rating"),
            "interpretation_statement": sensitivity_result.get("interpretation")
        }

    # 7. Causal Refutation & Falsification Tests
    if refutation_results and knowledge_source in [KNOWLEDGE_SOURCE_DATASET, KNOWLEDGE_SOURCE_HYBRID]:
        ref_summaries = []
        for ref in refutation_results:
            ref_summaries.append({
                "test_name": ref.get("test"),
                "passed": ref.get("passed"),
                "original_ate": ref.get("original_ate"),
                "refuted_ate": ref.get("mean_placebo_ate", ref.get("refuted_ate", ref.get("mean_subset_ate"))),
                "status": ref.get("status"),
                "interpretation": ref.get("explanation")
            })
        context["refutation_battery_results"] = ref_summaries

    # 8. Document Knowledge & Retrieved Passages (when documents or hybrid is active)
    if retrieved_passages and knowledge_source in [KNOWLEDGE_SOURCE_DOCUMENTS, KNOWLEDGE_SOURCE_HYBRID]:
        context["retrieved_document_evidence"] = [
            {
                "source_id": i + 1,
                "title": p.get("title", "Unknown"),
                "file_name": p.get("file_name", "Unknown"),
                "file_type": p.get("file_type", "doc").upper(),
                "page_or_unit": p.get("page_num", 1),
                "section": p.get("section", "Context"),
                "match_similarity": round(float(p.get("similarity_score", 1.0)), 3),
                "text": p.get("text", "")
            }
            for i, p in enumerate(retrieved_passages)
        ]
        # Extract numerical tokens from text passages so claim evaluation recognizes valid numbers
        passage_nums = []
        for p in retrieved_passages:
            passage_nums.extend(re.findall(r"[-+]?\b\d+(?:\.\d+)?", p.get("text", "")))
        if passage_nums:
            clean_nums = []
            for n in passage_nums[:60]:
                try:
                    clean_nums.append(round(float(n), 4))
                except ValueError:
                    pass
            context["document_numeric_signals"] = clean_nums

    if document_catalog and knowledge_source in [KNOWLEDGE_SOURCE_DOCUMENTS, KNOWLEDGE_SOURCE_HYBRID]:
        context["available_literature_catalog"] = [
            {
                "title": d.get("title", ""),
                "file_name": d.get("file_name", ""),
                "format": d.get("file_type", "").upper(),
                "units": d.get("total_pages", 1),
                "chunks": d.get("total_chunks", 1)
            }
            for d in document_catalog[:10]
        ]

    # 9. Missing Evidence & Known Limitations
    missing_elements = []
    if knowledge_source in [KNOWLEDGE_SOURCE_DATASET, KNOWLEDGE_SOURCE_HYBRID]:
        if not all_estimators:
            missing_elements.append("Causal estimators have not been executed on the active dataset yet.")
        if not sensitivity_result:
            missing_elements.append("Sensitivity E-value has not been computed.")
        if not refutation_results:
            missing_elements.append("Refutation battery (placebo, common cause, subset) has not been run.")
        missing_elements.append("Observational data cannot directly verify the No Unmeasured Confounding assumption.")
    if knowledge_source in [KNOWLEDGE_SOURCE_DOCUMENTS, KNOWLEDGE_SOURCE_HYBRID]:
        if not retrieved_passages:
            missing_elements.append("No relevant document passages were retrieved matching this inquiry.")
    context["missing_evidence_and_unobserved_limits"] = missing_elements

    return context


def build_system_prompt(mode: str = MODE_GROUNDED, knowledge_source: str = KNOWLEDGE_SOURCE_DATASET) -> str:
    """
    Construct the research-grade system prompt constraining Gemini to cite
    computed evidence, prevent hallucinations, communicate assumptions, and
    recommend scientific next steps across the chosen knowledge source.
    """
    if mode == MODE_AUDITOR:
        return f"""
You are a skeptical peer-review methodologist for a premier journal (JASA, Biometrika, Nature Human Behaviour).
Your role is to rigorously critique the observational study and literature evidence across knowledge source '{knowledge_source}':
1. Scrutinize baseline covariate imbalance using Rubin's criteria (SMD < 0.10, variance ratios 0.8-1.25).
2. Challenge the plausibility of unmeasured confounding using the computed E-value.
3. Review whether common support positivity was violated or if propensity trimming is required.
4. Scrutinize refutation test failures or passes.
5. Scrutinize preprints for methodological flaws, collider stratification bias, over-controlling via mediators, or selection bias.
6. Demand specific robustness checks and future experimental designs.
"""
    elif mode == MODE_UNCONSTRAINED:
        return """
You are an unconstrained AI research scientist. Answer the user's question about the dataset, literature preprints, and causal methodology based on general scientific knowledge. You may speculate and offer intuitive guidance without strictly citing computed evidence objects.
"""

    # MODE_GROUNDED with knowledge_source customization
    if knowledge_source == KNOWLEDGE_SOURCE_DOCUMENTS:
        return """
You are CausalLens AI's Literature & Document Intelligence Research Assistant.
Your mandate is to answer questions about uploaded research documents, preprints, reports, and slide decks (PDF, Word, Excel, CSV, PPTX, Markdown, Text).

CRITICAL GROUNDING & VERIFICATION RULES:
1. Ground every claim, methodology, sample size, or empirical finding directly in the provided document passages.
2. EVERY factual assertion MUST cite its source document title, format, and unit/page number using:
   [Source N: "<Document Title>", Type: <FORMAT>, Page/Unit: <PageNum>]
3. When asked to explain research papers:
   Break down the research objective, target population, structural identification assumptions, estimators used, and main findings.
4. When asked to summarize or compare multiple documents:
   Construct a clear comparative synthesis highlighting areas of methodological consensus vs. divergence, contrasting assumptions, and quantitative results.
5. If the provided excerpts do not contain sufficient evidence to answer the question, you MUST clearly state:
   "The ingested documents do not contain sufficient evidence to address this question."
6. NEVER invent citations, document titles, page numbers, or external literature.
"""
    elif knowledge_source == KNOWLEDGE_SOURCE_HYBRID:
        return """
You are CausalLens AI's Integrated Dual-Engine Causal & Literature Research Assistant.
You have simultaneous access to:
1. COMPUTED EMPIRICAL EVIDENCE from the active dataset (OLS, Matching, IPW, AIPW, DML, E-values, Refutations).
2. VERIFIED DOCUMENT EXCERPTS from ingested peer-reviewed preprints and research documents.

CRITICAL DUAL-GROUNDING RULES:
1. Synthesize both knowledge sources seamlessly: connect the dataset's computed numbers to the theoretical frameworks and assumptions established in the literature.
2. For computed dataset metrics, cite:
   [Evidence: <metric_name> = <exact_value>]
3. For document literature findings, cite:
   [Source N: "<Document Title>", Type: <FORMAT>, Page/Unit: <PageNum>]
4. Explain why computed estimates (e.g. OLS vs AIPW vs Double ML) agree or diverge, referencing literature mechanisms (e.g., Neyman orthogonality, cross-fitting, propensity model misspecification).
5. Never invent statistical results or external literature.
"""
    else:  # KNOWLEDGE_SOURCE_DATASET
        return """
You are CausalLens AI, an expert biostatistician and causal inference research assistant.
You are directly integrated into an advanced causal decision-intelligence platform.

YOUR CORE MANDATE:
Generate evidence-grounded, scientifically rigorous responses to the user's research questions regarding the active dataset and computed causal analytics.
You must strictly adhere to the following principles:

1. EVIDENCE CITATION (STRICT):
   Every quantitative claim, estimate, confidence interval, sample size, or test result MUST be cited
   directly from the supplied JSON evidence payload using the format:
   [Evidence: <metric_name> = <exact_value>]
   Example: "The adjusted ATE via Doubly Robust AIPW is 2.45 [Evidence: aipw_ate = 2.45] with a 95% CI of [1.80, 3.10] [Evidence: aipw_ci = [1.80, 3.10]]."

2. PREVENT HALLUCINATIONS & COMMUNICATE MISSING EVIDENCE:
   - If a requested statistic, variable, or test is NOT present in the computed evidence payload, you MUST explicitly state:
     "This metric is not available in the computed evidence."
   - NEVER fabricate numbers, guess sample sizes, or invent outside data.

3. CAUSALITY VS. CORRELATION:
   - NEVER state that an observational estimate "proves" causality or that a treatment "definitively caused" an outcome.
   - Always clarify that the adjusted estimate represents a causal parameter (ATE or ATT) ONLY under the validity of the structural identification assumptions.

4. EXPLICITLY STATE THE 5 CORE IDENTIFICATION ASSUMPTIONS:
   Whenever interpreting treatment effects, clearly discuss:
   a) Conditional Exchangeability / Unconfoundedness: No unmeasured confounders.
   b) Positivity / Common Support: All units have non-zero probability of both treatment states.
   c) SUTVA: No interference between units and no hidden treatment variations.
   d) Consistency: Observed outcome matches potential outcome for assigned treatment.
   e) Model Specification: Adequate functional form of regression and propensity models.

5. METHODOLOGICAL COMPARISONS:
   When asked to compare analytical methods (e.g., OLS vs. PSM vs. IPW vs. AIPW vs. DML):
   - Explain why estimates may diverge (e.g. OLS suffers from functional misspecification and extreme extrapolation; Matching focuses on the ATT and trims non-overlapping regions; IPW relies on propensity calibration; Doubly Robust AIPW is consistent if either outcome or propensity model holds; DML uses cross-fitting to eliminate machine learning regularization bias).

6. IDENTIFY CONFOUNDING & SENSITIVITY:
   - Cite the baseline SMDs and post-matching balance tables.
   - Interpret the E-value: state the minimum confounder association required to explain away the effect.
   - Summarize the refutation test battery results (placebo treatment, random common cause, subset stability).

7. RECOMMEND SCIENTIFIC NEXT STEPS:
   Provide actionable recommendations: prospective randomized controlled trials (RCTs), instrumental variable (IV) exploration, negative control outcomes, stratified subgroup analysis, or non-linear sensitivity bounds.
"""


def answer_with_evidence(
    question: str,
    context: Dict[str, Any],
    api_key: Any = _ENV_DEFAULT,
    mode: str = MODE_GROUNDED,
    knowledge_source: str = KNOWLEDGE_SOURCE_DATASET,
    model_name: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    vector_store: Optional[Any] = None,
    top_k: int = 5,
    paper_filter: Optional[str] = None,
    file_type_filter: Optional[str] = None,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Generate an evidence-grounded research response with Google Gemini (or offline fallback)
    and evaluate its numerical faithfulness and assumption disclosure rate across
    the selected knowledge source (Structured Dataset, Uploaded Documents, or Hybrid).
    """
    citations = []
    retrieved_passages = []

    # 1. Retrieve document passages if querying documents or hybrid mode
    if vector_store is not None and knowledge_source in [KNOWLEDGE_SOURCE_DOCUMENTS, KNOWLEDGE_SOURCE_HYBRID]:
        try:
            retrieved_passages = vector_store.search_documents(
                query=question,
                top_k=top_k,
                paper_filter=paper_filter,
                file_type_filter=file_type_filter,
                api_key=api_key
            )
            for p in retrieved_passages:
                citations.append({
                    "paper": p.get("title", "Unknown"),
                    "file": p.get("file_name", "Unknown"),
                    "file_type": p.get("file_type", "doc"),
                    "page": p.get("page_num", 1),
                    "section": p.get("section", "Context"),
                    "excerpt": p.get("text", "")[:250],
                    "similarity_score": p.get("similarity_score", 1.0)
                })

            context["retrieved_document_evidence"] = [
                {
                    "source_id": i + 1,
                    "title": p.get("title", "Unknown"),
                    "file_name": p.get("file_name", "Unknown"),
                    "file_type": p.get("file_type", "doc").upper(),
                    "page_or_unit": p.get("page_num", 1),
                    "section": p.get("section", "Context"),
                    "match_similarity": round(float(p.get("similarity_score", 1.0)), 3),
                    "text": p.get("text", "")
                }
                for i, p in enumerate(retrieved_passages)
            ]

            p_nums = []
            for p in retrieved_passages:
                p_nums.extend(re.findall(r"[-+]?\b\d+(?:\.\d+)?", p.get("text", "")))
            if p_nums:
                clean_nums = []
                for n in p_nums[:60]:
                    try:
                        clean_nums.append(round(float(n), 4))
                    except ValueError:
                        pass
                context["document_numeric_signals"] = clean_nums
        except Exception:
            pass

    if api_key is _ENV_DEFAULT:
        key = get_api_key()
    else:
        key = api_key

    # 2. Offline Fallback Synthesis when API key is not configured
    if not key:
        fallback_lines = []
        if knowledge_source == KNOWLEDGE_SOURCE_DOCUMENTS:
            if retrieved_passages:
                fallback_lines.append("**Offline Document Intelligence Synthesis (API Key Not Configured):**\n")
                fallback_lines.append("Top retrieved passages from ChromaDB & BM25Okapi vector index:\n")
                for i, p in enumerate(retrieved_passages, 1):
                    ft = p.get("file_type", "doc").upper()
                    sim = p.get("similarity_score", 1.0)
                    fallback_lines.append(f"**[{i}] [{ft}] {p.get('title', 'Unknown')} (Page/Unit {p.get('page_num', 1)})** — Match: `{sim:.1%}` | Section: `{p.get('section', 'Context')}`")
                    fallback_lines.append(f"> {p.get('text', '')[:320]}...\n")
            else:
                fallback_lines.append("The ingested documents do not contain sufficient evidence to address this question (no relevant passages found in vector store).")
        
        elif knowledge_source == KNOWLEDGE_SOURCE_DATASET:
            d_prof = context.get("dataset_profile", {})
            c_spec = context.get("causal_specification", {})
            ests = context.get("causal_estimators_evaluated", [])
            sens = context.get("sensitivity_analysis_e_value", {})
            
            fallback_lines.append(f"**Offline Dataset Causal Analytics (API Key Not Configured):**\n")
            if d_prof:
                fallback_lines.append(f"- **Active Dataset:** `{d_prof.get('name')}` with **{d_prof.get('total_observations')} observations** and **{d_prof.get('feature_count')} features**.")
            if c_spec:
                fallback_lines.append(f"- **Causal Specification:** Treatment = `{c_spec.get('treatment_variable')}`, Outcome = `{c_spec.get('outcome_variable')}`, Confounders = `{', '.join(c_spec.get('adjusted_confounders', []))}`.")
            
            if ests:
                fallback_lines.append("\n**Computed Causal Estimators (ATE & ATT):**")
                for e in ests:
                    ate_str = f"{e['ate_point_estimate']:.4f}" if e.get("ate_point_estimate") is not None else "N/A"
                    ci_str = f"[{e['ate_95_ci'][0]:.4f}, {e['ate_95_ci'][1]:.4f}]" if e.get("ate_95_ci") else "N/A"
                    se_str = f"{e['ate_std_error']:.4f}" if e.get("ate_std_error") is not None else "N/A"
                    p_str = f"{e['ate_p_value']:.4f}" if e.get("ate_p_value") is not None else "N/A"
                    fallback_lines.append(f"- **{e['estimator']}**: ATE = `{ate_str}` (SE = `{se_str}`, 95% CI = `{ci_str}`, p = `{p_str}`) [Evidence: {e['estimator'].lower()}_ate = {ate_str}]")
            
            if sens:
                fallback_lines.append(f"\n- **Sensitivity Analysis:** E-Value point estimate = `{sens.get('e_value_point_estimate')}` (CI bound = `{sens.get('e_value_ci_bound')}`). Robustness: **{sens.get('robustness_rating')}**.")
            
            fallback_lines.append("\n*Note: To run generative synthesis, connect your Google Gemini API key in the sidebar.*")

        else:  # HYBRID
            fallback_lines.append("**Offline Hybrid Dual-Source Intelligence (API Key Not Configured):**\n")
            ests = context.get("causal_estimators_evaluated", [])
            if ests:
                fallback_lines.append("**Empirical Dataset Results:**")
                for e in ests[:3]:
                    fallback_lines.append(f"- **{e['estimator']}**: ATE = `{e.get('ate_point_estimate')}` (95% CI: `{e.get('ate_95_ci')}`)")
            if retrieved_passages:
                fallback_lines.append("\n**Top Retrieved Document Passages:**")
                for p in retrieved_passages[:2]:
                    fallback_lines.append(f"- **[{p.get('file_type', 'doc').upper()}] {p.get('title')} (Unit {p.get('page_num')})**: {p.get('text', '')[:220]}...")

        fallback_text = "\n".join(fallback_lines)
        faithfulness = verify_claims_against_evidence(fallback_text, context)
        assumptions_audit = audit_causal_assumptions(fallback_text)

        return {
            "text": fallback_text,
            "mode": mode,
            "knowledge_source": knowledge_source,
            "model": "local-fallback",
            "faithfulness": faithfulness,
            "assumptions_audit": assumptions_audit,
            "citations": citations,
            "retrieved_passages": retrieved_passages,
            "error": "MISSING_API_KEY"
        }

    # 3. Online Gemini API Synthesis
    selected_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # Serialize structured context safely
    context_str = json.dumps(context, default=str, indent=2)
    if len(context_str) > 75000:
        context_str = context_str[:75000] + "\n... [Context truncated for length] ...}"

    system_instruction = build_system_prompt(mode, knowledge_source=knowledge_source)

    # Multi-turn history formatting if available
    history_str = ""
    if history:
        history_str = "\nPREVIOUS CONVERSATION CONTEXT:\n"
        for turn in history[-4:]:
            history_str += f"User: {turn.get('question', '')}\nAssistant: {turn.get('response', '')[:400]}...\n"

    prompt = f"""
{system_instruction}

{history_str}

COMPUTED EMPIRICAL EVIDENCE & RETRIEVED SOURCES (JSON):
```json
{context_str}
```

USER RESEARCH QUESTION:
{question}
"""

    gem_res = generate_gemini_text(
        prompt=prompt,
        system_instruction=system_instruction,
        preferred_model=selected_model,
        temperature=temperature,
        api_key=key,
    )

    if gem_res["success"]:
        response_text = gem_res["text"] or "The model returned an empty response."
        used_model = gem_res.get("model_used", selected_model)

        # Compute empirical faithfulness & assumption disclosure metrics
        faithfulness = verify_claims_against_evidence(response_text, context)
        assumptions_audit = audit_causal_assumptions(response_text)

        return {
            "text": response_text,
            "mode": mode,
            "knowledge_source": knowledge_source,
            "model": used_model,
            "faithfulness": faithfulness,
            "assumptions_audit": assumptions_audit,
            "citations": citations,
            "retrieved_passages": retrieved_passages,
            "error": None
        }
    else:
        clean_err = sanitize_message(gem_res.get("error", "LLM request failed"), key)
        err_msg = (
            f"**LLM request failed:** {clean_err}\n\n"
            "Your computed statistical results, balance diagnostics, and sensitivity tests remain fully valid above."
        )
        return {
            "text": err_msg,
            "mode": mode,
            "knowledge_source": knowledge_source,
            "model": selected_model,
            "faithfulness": {"faithfulness_score": 100.0, "total_claims": 0, "grounded_claims": 0},
            "assumptions_audit": {"disclosed_count": 0, "total_assumptions": 5, "rigor_rating": "Error"},
            "citations": citations,
            "retrieved_passages": retrieved_passages,
            "error": clean_err
        }
