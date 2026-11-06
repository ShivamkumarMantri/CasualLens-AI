import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.causal import estimate_ate


def compute_e_value(
    ate: float,
    se: float,
    outcome_sd: float,
    ci_low: float = None,
    ci_high: float = None
) -> Dict[str, Any]:
    """
    Compute the E-value for a continuous outcome (VanderWeele & Ding, Ann Intern Med 2017).
    The E-value represents the minimum risk ratio that an unmeasured confounder would
    need to have with both the treatment and the outcome, conditional on measured covariates,
    to explain away the observed causal effect.
    
    Uses the standardized mean difference (Cohen's d) conversion to approximate risk ratio:
        d = ate / outcome_sd
        RR ~ exp(0.91 * d)
        E-value = RR + sqrt(RR * (RR - 1))  [for RR >= 1]
    """
    if outcome_sd <= 1e-9:
        outcome_sd = 1.0

    d = ate / outcome_sd
    # Convert standardized effect d to approximate Risk Ratio
    # VanderWeele & Ding approximation: RR = exp(0.91 * d)
    rr_point = float(np.exp(0.91 * abs(d)))
    
    if rr_point >= 1.0:
        e_value_point = float(rr_point + np.sqrt(rr_point * (rr_point - 1.0)))
    else:
        e_value_point = 1.0

    # For confidence interval closest to the null
    e_value_ci = 1.0
    if ci_low is not None and ci_high is not None:
        # If CI covers 0, E-value for CI is 1.0 (already explainable by chance)
        if ci_low <= 0 <= ci_high:
            e_value_ci = 1.0
        else:
            # Bound closest to null
            closest_bound = ci_low if ate > 0 else abs(ci_high)
            d_ci = closest_bound / outcome_sd
            rr_ci = float(np.exp(0.91 * abs(d_ci)))
            if rr_ci >= 1.0:
                e_value_ci = float(rr_ci + np.sqrt(rr_ci * (rr_ci - 1.0)))

    # Qualitative robustness interpretation
    if e_value_point >= 3.0:
        robustness = "High robustness to unmeasured confounding"
    elif e_value_point >= 1.5:
        robustness = "Moderate robustness; plausible unmeasured confounders could explain the effect"
    else:
        robustness = "Fragile; small unmeasured confounding could explain away the effect"

    return {
        "e_value_point": round(e_value_point, 3),
        "e_value_ci": round(e_value_ci, 3),
        "standardized_effect_d": round(float(d), 3),
        "approx_risk_ratio": round(rr_point, 3),
        "robustness_rating": robustness,
        "interpretation": (
            f"An unmeasured confounder must have a risk ratio of at least {e_value_point:.2f} "
            f"with both the treatment and outcome (above and beyond measured covariates) to reduce the effect to null. "
            f"To shift the 95% CI to include the null, an unmeasured confounder requires an association of at least {e_value_ci:.2f}."
        )
    }


def compute_dual_e_values(
    ate: float,
    att: float,
    outcome_sd: float,
    ate_ci: tuple = (None, None),
    att_ci: tuple = (None, None)
) -> Dict[str, Any]:
    """
    Compute E-values for both the Average Treatment Effect (ATE) and
    the Average Treatment Effect on the Treated (ATT).
    """
    ev_ate = compute_e_value(ate=ate, se=0.0, outcome_sd=outcome_sd, ci_low=ate_ci[0], ci_high=ate_ci[1])
    ev_att = compute_e_value(ate=att, se=0.0, outcome_sd=outcome_sd, ci_low=att_ci[0], ci_high=att_ci[1])

    return {
        "ate_e_value": ev_ate,
        "att_e_value": ev_att,
        "e_value_point": ev_ate["e_value_point"],
        "e_value_ci": ev_ate["e_value_ci"],
        "robustness_rating": ev_ate["robustness_rating"],
        "interpretation": ev_ate["interpretation"],
    }


def run_placebo_treatment_refutation(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list,
    n_simulations: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Placebo Treatment Refutation Test:
    Randomly permute the treatment vector to break the causal connection.
    A valid causal estimator should yield an estimated effect near zero.
    """
    rng = np.random.default_rng(random_state)
    simulated_ates = []
    
    # Original estimate for reference
    orig = estimate_ate(df, treatment, outcome, covariates)
    orig_ate = orig["ate"]

    temp_df = df.copy()
    for _ in range(n_simulations):
        temp_df["_placebo_treatment"] = rng.permutation(temp_df[treatment].to_numpy())
        res = estimate_ate(temp_df, "_placebo_treatment", outcome, covariates)
        simulated_ates.append(res["ate"])

    mean_placebo_ate = float(np.mean(simulated_ates))
    std_placebo_ate = float(np.std(simulated_ates))

    # Passed if mean placebo effect is close to 0 and distinct from original effect
    passed = abs(mean_placebo_ate) < (0.25 * abs(orig_ate) + 0.1)

    return {
        "test": "Placebo Treatment Permutation Refutation",
        "original_ate": round(orig_ate, 4),
        "mean_placebo_ate": round(mean_placebo_ate, 4),
        "std_placebo_ate": round(std_placebo_ate, 4),
        "n_simulations": n_simulations,
        "simulated_ates": [round(x, 4) for x in simulated_ates],
        "passed": bool(passed),
        "status": "PASSED (Placebo effect collapsed to near zero)" if passed else "FAILED (Spurious signal detected on permuted treatment)",
        "explanation": "Replacing the true treatment with random permutation should abolish the estimated effect. Failure indicates model overfitting or structural bias."
    }


def run_random_common_cause_refutation(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Random Common Cause Refutation Test:
    Inject an independent random Gaussian variable into the covariate set.
    The true causal estimate should remain invariant to an irrelevant common cause.
    """
    rng = np.random.default_rng(random_state)
    orig = estimate_ate(df, treatment, outcome, covariates)
    orig_ate = orig["ate"]

    temp_df = df.copy()
    temp_df["_random_common_cause"] = rng.normal(0, 1, len(temp_df))
    augmented_covs = list(covariates) + ["_random_common_cause"]

    refuted = estimate_ate(temp_df, treatment, outcome, augmented_covs)
    refuted_ate = refuted["ate"]

    pct_change = abs(refuted_ate - orig_ate) / (abs(orig_ate) + 1e-6) * 100
    passed = pct_change < 10.0  # Should shift by less than 10%

    return {
        "test": "Random Common Cause Invariance Refutation",
        "original_ate": round(orig_ate, 4),
        "refuted_ate": round(refuted_ate, 4),
        "percent_change": round(pct_change, 2),
        "passed": bool(passed),
        "status": "PASSED (Estimate invariant to random noise confounder)" if passed else "FAILED (Estimate shifted significantly after adding noise)",
        "explanation": "Adding an independent random variable should not substantially perturb the estimated causal effect."
    }


def run_data_subset_refutation(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list,
    subset_fraction: float = 0.8,
    n_iterations: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Data Subset Refutation Test:
    Re-estimate the causal effect on random subsets (e.g. 80%) of the data.
    Assesses estimator stability and sensitivity to specific sample subsets.
    """
    rng = np.random.default_rng(random_state)
    orig = estimate_ate(df, treatment, outcome, covariates)
    orig_ate = orig["ate"]

    subset_ates = []
    n = len(df)
    subset_size = int(n * subset_fraction)

    for i in range(n_iterations):
        indices = rng.choice(n, size=subset_size, replace=False)
        sub_df = df.iloc[indices]
        try:
            res = estimate_ate(sub_df, treatment, outcome, covariates)
            subset_ates.append(res["ate"])
        except Exception:
            continue

    if not subset_ates:
        return {"test": "Data Subset Refutation", "passed": False, "status": "Failed to converge on subsets"}

    mean_subset_ate = float(np.mean(subset_ates))
    std_subset_ate = float(np.std(subset_ates))

    pct_drift = abs(mean_subset_ate - orig_ate) / (abs(orig_ate) + 1e-6) * 100
    passed = pct_drift < 15.0

    return {
        "test": f"Data Subset Stability Refutation ({int(subset_fraction*100)}% Subsamples)",
        "original_ate": round(orig_ate, 4),
        "mean_subset_ate": round(mean_subset_ate, 4),
        "std_subset_ate": round(std_subset_ate, 4),
        "percent_drift": round(pct_drift, 2),
        "n_successful_runs": len(subset_ates),
        "passed": bool(passed),
        "status": "PASSED (Stable across random subsamples)" if passed else "FAILED (High sensitivity to sample exclusion)",
        "explanation": "Estimator stability across random data subsets demonstrates robustness against influence of outliers or specific clusters."
    }

