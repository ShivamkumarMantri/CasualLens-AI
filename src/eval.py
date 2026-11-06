import re
from typing import Dict, Any, List, Set


def _collect_numbers_from_dict(obj: Any, collected: Set[float]):
    """Recursively extract all numeric values from a nested dictionary/list structure."""
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        # Filter out NaN or infinite
        if obj == obj and abs(obj) != float("inf"):
            collected.add(round(float(obj), 4))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers_from_dict(v, collected)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_numbers_from_dict(item, collected)


def extract_numerical_claims(text: str) -> List[float]:
    """
    Extract numerical claims from LLM response text.
    Handles floats, integers, percentages, negative numbers, and intervals.
    Filters out common boilerplate years/dates and nominal confidence levels (e.g., '95% CI').
    """
    # Filter out nominal confidence intervals phrases like '95% CI' or '95% confidence interval'
    # so they are not misclassified as empirical data claims
    sanitized_text = re.sub(r"\b(?:90|95|99)%\s*(?:ci|confidence\s+interval)?\b", " ", text, flags=re.IGNORECASE)

    pattern = r"[-+]?\b\d+(?:\.\d+)?%?"
    raw_matches = re.findall(pattern, sanitized_text)
    numbers = []
    
    for m in raw_matches:
        is_pct = m.endswith("%")
        cleaned = m.rstrip("%")
        try:
            val = float(cleaned)
            # Exclude standard year references (e.g. 1990-2030) unless decimal
            if "." not in cleaned and 1990 <= val <= 2030 and not is_pct:
                continue
            numbers.append(round(val, 4))
            if is_pct:
                # Also include proportional decimal representation (e.g. 25% -> 0.25)
                numbers.append(round(val / 100.0, 4))
        except ValueError:
            continue
            
    return numbers


def verify_claims_against_evidence(
    response_text: str,
    context: Dict[str, Any],
    rel_tolerance: float = 0.06,
    abs_tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Empirically verify whether numerical claims in the LLM response are supported by
    the computed evidence payload.
    
    Returns:
        total_claims: Number of extracted numeric claims
        grounded_claims: Number of claims matching evidence within tolerance
        unsupported_claims: Number of claims with no match in computed evidence
        faithfulness_score: Percentage of grounded claims (0-100%)
        claims_detail: List of verification records for inspection
    """
    evidence_numbers = set()
    _collect_numbers_from_dict(context, evidence_numbers)

    cited_numbers = extract_numerical_claims(response_text)
    if not cited_numbers:
        return {
            "total_claims": 0,
            "grounded_claims": 0,
            "unsupported_claims": 0,
            "faithfulness_score": 100.0,
            "claims_detail": [],
            "grounding_status": "No numerical claims made in response."
        }

    claims_detail = []
    grounded_count = 0

    for num in cited_numbers:
        # Check if num matches any number in evidence within relative or absolute tolerance
        matched = False
        closest_diff = float("inf")
        closest_evidence = None

        for ev in evidence_numbers:
            diff = abs(num - ev)
            scale = max(abs(num), abs(ev), 1.0)
            rel_diff = diff / scale
            if diff < closest_diff:
                closest_diff = diff
                closest_evidence = ev

            if diff <= abs_tolerance or rel_diff <= rel_tolerance:
                matched = True
                break

        if matched:
            grounded_count += 1
            claims_detail.append({
                "claimed_value": num,
                "status": "VERIFIED",
                "matched_evidence": closest_evidence
            })
        else:
            claims_detail.append({
                "claimed_value": num,
                "status": "UNSUPPORTED",
                "nearest_evidence": closest_evidence,
                "difference": round(closest_diff, 4) if closest_diff != float("inf") else None
            })

    unsupported_count = len(cited_numbers) - grounded_count
    faithfulness_score = round(grounded_count / len(cited_numbers) * 100.0, 1)

    return {
        "total_claims": len(cited_numbers),
        "grounded_claims": grounded_count,
        "unsupported_claims": unsupported_count,
        "faithfulness_score": faithfulness_score,
        "claims_detail": claims_detail,
        "grounding_status": (
            "Fully Grounded (100%)" if faithfulness_score == 100.0
            else f"Substantially Grounded ({faithfulness_score}%)" if faithfulness_score >= 80.0
            else f"Partially Grounded ({faithfulness_score}%)"
        )
    }


def audit_causal_assumptions(response_text: str) -> Dict[str, Any]:
    """
    Audit whether the assistant text appropriately discloses the fundamental identification
    assumptions and avoids ungrounded causal overclaiming.
    """
    text_lower = response_text.lower()

    assumptions = {
        "unconfoundedness": any(k in text_lower for k in [
            "unconfounded", "unmeasured confound", "ignorability", "exchangeab", "omitted variable", "backdoor"
        ]),
        "positivity": any(k in text_lower for k in [
            "positivity", "overlap", "common support"
        ]),
        "sutva": any(k in text_lower for k in [
            "sutva", "interference", "stable unit", "spillover"
        ]),
        "consistency": any(k in text_lower for k in [
            "consistency", "treatment definition", "multiple versions"
        ]),
        "model_specification": any(k in text_lower for k in [
            "misspecif", "functional form", "linearity", "model adequate", "specification"
        ]),
    }

    disclosed_count = sum(assumptions.values())
    disclosure_rate_pct = round(disclosed_count / len(assumptions) * 100.0, 1)

    # Check for overclaiming causal language
    overclaiming_triggers = [
        "proves that", "definitively proves", "undeniably caused", "conclusively demonstrates causality",
        "guarantees that", "direct proof"
    ]
    detected_overclaims = [phrase for phrase in overclaiming_triggers if phrase in text_lower]

    return {
        "disclosed_count": disclosed_count,
        "total_assumptions": len(assumptions),
        "disclosure_rate_pct": disclosure_rate_pct,
        "assumptions_detail": assumptions,
        "has_overclaiming": len(detected_overclaims) > 0,
        "detected_overclaims": detected_overclaims,
        "rigor_rating": (
            "Exemplary Research Standard" if disclosed_count >= 3 and not detected_overclaims
            else "Acceptable Disclosure" if disclosed_count >= 1 and not detected_overclaims
            else "Methodologically Deficient"
        )
    }
