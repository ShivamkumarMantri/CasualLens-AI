import json
from datetime import datetime, timezone
import pandas as pd
from typing import Dict, Any, List, Optional


def _format_num(val: Any, fmt: str = ".4f", default: str = "N/A") -> str:
    """Safely format a numerical value without raising TypeError on None."""
    if val is None or (isinstance(val, float) and (val != val or abs(val) == float("inf"))):
        return default
    try:
        return f"{float(val):{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def build_markdown_report(
    df: pd.DataFrame,
    causal_result: Optional[Dict[str, Any]] = None,
    analysis_spec: Optional[Dict[str, Any]] = None,
    all_estimators: Optional[List[Dict[str, Any]]] = None,
    balance_table: Optional[List[Dict[str, Any]]] = None,
    sensitivity_result: Optional[Dict[str, Any]] = None,
    refutation_results: Optional[List[Dict[str, Any]]] = None,
    dag_summary: Optional[Dict[str, Any]] = None,
    assistant_logs: Optional[List[Dict[str, Any]]] = None,
    literature_summary: Optional[Dict[str, Any]] = None
) -> str:
    """
    Construct a research-level, publication-grade scientific Markdown report.
    Safely handles missing or partial analysis components.
    """
    spec = analysis_spec or {}
    treatment = spec.get("treatment", "Unspecified")
    outcome = spec.get("outcome", "Unspecified")
    confounders = spec.get("confounders", [])

    lines = [
        "# CausalLens AI — Research-Grade Observational Analysis Report",
        f"\n**Generated (UTC):** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  ",
        "**System:** CausalLens AI (Research Decision-Intelligence Platform)  ",
        f"**Target Estimand:** Average Treatment Effect (ATE) / Average Treatment Effect on the Treated (ATT) of `{treatment}` on `{outcome}`  ",
        "\n---",
        "\n## 1. Executive Summary",
        "This report documents an evidence-grounded observational causal inference study. "
        "In accordance with rigorous methodological standards, statistical associations are distinguished "
        "from causal effects via explicit identification assumptions (consistency, positivity, unconfoundedness, "
        "no interference, and functional model adequacy).",
        "\n## 2. Dataset Overview and Quality Audit",
        f"- **Sample Size (N):** {len(df):,}",
        f"- **Features / Columns:** {len(df.columns)}",
        f"- **Total Missing Cells:** {int(df.isna().sum().sum()):,} ({df.isna().mean().mean()*100:.2f}%)",
        f"- **Duplicate Rows:** {int(df.duplicated().sum()):,}",
        "\n### Variable Metadata",
        "\n" + pd.DataFrame({
            "Feature": df.columns,
            "Dtype": [str(x) for x in df.dtypes],
            "Missing": [int(df[c].isna().sum()) for c in df.columns],
            "Unique Values": [int(df[c].nunique()) for c in df.columns]
        }).to_markdown(index=False),
        "\n---",
        "\n## 3. Causal Identification Strategy & Structural Assumptions",
        f"- **Treatment Variable (T):** `{treatment}`",
        f"- **Outcome Variable (Y):** `{outcome}`",
        f"- **Adjusted Covariate Set (X):** {', '.join([f'`{c}`' for c in confounders]) if confounders else 'None'}",
    ]

    # DAG Identification section
    if dag_summary:
        lines += [
            f"- **Backdoor Criterion Status:** {dag_summary.get('identification_status', 'Evaluated')}",
            f"- **Sufficient Adjustment Set:** {', '.join(dag_summary.get('sufficient_adjustment_set', []))}",
        ]
        if dag_summary.get("warnings"):
            lines.append("\n> [!WARNING]\n> " + "\n> ".join(dag_summary.get("warnings", [])))

    lines += [
        "\n### Core Identification Assumptions:",
        "1. **Conditional Exchangeability (Unconfoundedness):** $Y(1), Y(0) \\perp T \\mid X$. No unmeasured confounders exist.",
        "2. **Positivity (Common Support):** $0 < P(T=1 \\mid X) < 1$ for all values of $X$ in the population.",
        "3. **Stable Unit Treatment Value Assumption (SUTVA):** No interference between units and no hidden variations of treatment.",
        "4. **Consistency:** If $T_i = t$, then the observed outcome equals the potential outcome $Y_i(t)$.",
        "\n---",
        "\n## 4. Causal Estimators and Benchmarking",
    ]

    # Comparative Estimators Table
    if all_estimators:
        est_rows = []
        for r in all_estimators:
            ci_str = f"[{_format_num(r.get('ci_low'))}, {_format_num(r.get('ci_high'))}]" if "ci_low" in r else "N/A"
            att_ci_str = f"[{_format_num(r.get('att_ci_low'))}, {_format_num(r.get('att_ci_high'))}]" if "att_ci_low" in r else "N/A"
            est_rows.append({
                "Estimator": r.get("estimator", r.get("method", "Unknown")),
                "Point Estimate (ATE)": _format_num(r.get("ate")),
                "95% CI (ATE)": ci_str,
                "Point Estimate (ATT)": _format_num(r.get("att")),
                "95% CI (ATT)": att_ci_str,
                "Std. Error": _format_num(r.get("se")),
                "p-value": _format_num(r.get("p_value"), fmt=".5g"),
                "Sample (N)": r.get("n", r.get("n_pairs", "N/A"))
            })
        lines.append("\n" + pd.DataFrame(est_rows).to_markdown(index=False))
    elif causal_result:
        ci_str = f"[{_format_num(causal_result.get('ci_low'))}, {_format_num(causal_result.get('ci_high'))}]"
        lines += [
            f"- **Estimator:** {causal_result.get('method', 'OLS outcome regression')}",
            f"- **Point Estimate (ATE):** {_format_num(causal_result.get('ate'))}",
            f"- **Standard Error:** {_format_num(causal_result.get('se', causal_result.get('diagnostics', {}).get('robust_se')))}",
            f"- **95% Confidence Interval:** {ci_str}",
            f"- **p-value:** {_format_num(causal_result.get('p_value'), fmt='.5g')}",
        ]
    else:
        lines.append("No causal estimate has been executed.")

    # Covariate Balance Section
    if balance_table:
        lines += [
            "\n---",
            "\n## 5. Covariate Balance & Overlap Diagnostics",
            "Covariate balance assesses whether adjustment successfully eliminated baseline confounding.",
            "According to Rubin's criteria, a Standardized Mean Difference (SMD) $< 0.10$ signifies acceptable balance.",
            "\n" + pd.DataFrame(balance_table).to_markdown(index=False)
        ]

    # Sensitivity Analysis Section
    if sensitivity_result:
        lines += [
            "\n---",
            "\n## 6. Sensitivity Analysis (E-Value)",
            f"- **Point Estimate E-Value:** `{sensitivity_result.get('e_value_point')}`",
            f"- **95% CI E-Value:** `{sensitivity_result.get('e_value_ci')}`",
            f"- **Robustness Assessment:** {sensitivity_result.get('robustness_rating')}",
            f"\n> **Methodological Interpretation:** {sensitivity_result.get('interpretation')}"
        ]

    # Refutation Tests Section
    if refutation_results:
        lines += [
            "\n---",
            "\n## 7. Causal Refutation & Placebo Tests",
            "Refutation tests probe the falsifiability of the causal model.",
        ]
        ref_rows = []
        for ref in refutation_results:
            ref_rows.append({
                "Test": ref.get("test"),
                "Original ATE": ref.get("original_ate"),
                "Refuted ATE": ref.get("mean_placebo_ate", ref.get("refuted_ate", ref.get("mean_subset_ate"))),
                "Status": ref.get("status"),
                "Passed": "YES" if ref.get("passed") else "NO"
            })
        lines.append("\n" + pd.DataFrame(ref_rows).to_markdown(index=False))

    # Assistant & Faithfulness Audit
    if assistant_logs:
        lines += [
            "\n---",
            "\n## 8. AI Research Assistant Audit & Faithfulness Verification",
            "The integrated GenAI assistant was interrogated under constrained research prompting.",
        ]
        for idx, log in enumerate(assistant_logs, 1):
            f_score = log.get("faithfulness", {}).get("faithfulness_score", "N/A")
            asmp_score = log.get("assumptions_audit", {}).get("disclosed_count", "N/A")
            lines += [
                f"\n### Query {idx}: {log.get('question')}",
                f"**Faithfulness Score:** `{f_score}%` | **Assumptions Disclosed:** `{asmp_score}/5` | **Mode:** `{log.get('mode', 'grounded')}`",
                f"\n```text\n{log.get('response', '')}\n```"
            ]

    if literature_summary:
        lines += [
            "\n---",
            "\n## 9. Academic Literature & Grounded Citations (RAG)",
            f"**Indexed Papers:** {literature_summary.get('papers_count', 0)} | **Vector Collection Chunks:** {literature_summary.get('chunks_count', 0)}",
        ]
        papers = literature_summary.get("papers", [])
        if papers:
            lines += ["\n### Ingested Research Papers"]
            for p in papers:
                lines.append(f"- **{p.get('title', 'Unknown')}** ({p.get('file_name', '')}) — Pages: {p.get('total_pages', 'N/A')}, Chunks: {p.get('total_chunks', 'N/A')}")
        if literature_summary.get("recent_qa"):
            lines += [
                "\n### Grounded Research Queries & Citations",
                f"**Query:** {literature_summary['recent_qa'].get('query', '')}",
                f"**Synthesized Finding:**\n{literature_summary['recent_qa'].get('answer', '')}"
            ]

    lines += [
        "\n---",
        "\n## 10. Limitations & Threats to Validity",
        "1. **Unmeasured Confounding:** No observational statistical estimator can fully rule out bias from omitted confounders.",
        "2. **Functional Misspecification:** Non-linear interactions or misspecified propensity/outcome equations may distort parametric estimates.",
        "3. **Positivity Violations:** Sparse overlap in extreme covariate regions can inflate estimator variance.",
        "4. **Generalizability / Transportability:** Results apply only to the sampled target population and setting.",
        "\n## 11. Foundational Scientific References",
        "- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.",
        "- Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688-701.",
        "- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.",
        "- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. *JASA*, 89(427), 846-866.",
        "- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.",
        "- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.",
        "\n## 11. Reproducibility & Ethics Statement",
        "This analysis was produced using deterministically tracked parameters and open-source scientific software. "
        "Direct personal identifiers must be removed prior to external distribution, and all model findings must be cross-verified "
        "against the computed empirical tables before influencing clinical or operational decisions."
    ]

    return "\n".join(lines)


def build_json_reproducibility_bundle(
    df: pd.DataFrame,
    analysis_spec: Dict[str, Any],
    all_estimators: List[Dict[str, Any]],
    balance_table: Optional[List[Dict[str, Any]]] = None,
    sensitivity_result: Optional[Dict[str, Any]] = None,
    refutation_results: Optional[List[Dict[str, Any]]] = None,
    literature_summary: Optional[Dict[str, Any]] = None
) -> str:
    """
    Produce a complete, machine-readable JSON reproducibility bundle
    containing all specifications, parameters, and computed results.
    """
    bundle = {
        "metadata": {
            "platform": "CausalLens AI",
            "version": "2.0.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_observations": len(df),
            "columns": list(df.columns),
        },
        "analysis_spec": analysis_spec,
        "estimators": all_estimators,
        "covariate_balance": balance_table,
        "sensitivity": sensitivity_result,
        "refutation_tests": refutation_results,
        "literature_summary": literature_summary
    }
    return json.dumps(bundle, default=str, indent=2)


def build_html_report(markdown_content: str) -> str:
    """
    Wrap the generated markdown report into a standalone, styled HTML document
    with typography optimized for screen viewing and print-to-PDF export.
    """
    # Replace markdown headings with HTML for clean rendering
    html_lines = []
    for line in markdown_content.split("\n"):
        if line.startswith("# "):
            html_lines.append(f"<h1 style='color:#1565C0; border-bottom:2px solid #1565C0; padding-bottom:8px;'>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2 style='color:#2E7D32; margin-top:24px;'>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3 style='color:#455A64; margin-top:16px;'>{line[4:]}</h3>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.strip() == "---":
            html_lines.append("<hr style='border:0; border-top:1px solid #E0E0E0; margin:24px 0;'/>")
        else:
            html_lines.append(f"<p style='line-height:1.6;'>{line}</p>")

    body = "\n".join(html_lines)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CausalLens AI — Research Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            color: #212121;
            background-color: #FAFAFA;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            background: #fff;
        }}
        th, td {{
            padding: 10px 14px;
            border: 1px solid #E0E0E0;
            text-align: left;
            font-size: 14px;
        }}
        th {{
            background-color: #ECEFF1;
            color: #37474F;
            font-weight: 600;
        }}
        code {{
            background-color: #ECEFF1;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: Consolas, monospace;
            font-size: 13px;
        }}
        @media print {{
            body {{ background: #fff; max-width: 100%; margin: 20px; }}
        }}
    </style>
</head>
<body>
    {body}
</body>
</html>"""
