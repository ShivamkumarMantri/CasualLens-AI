import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.causal import (
    estimate_ate,
    propensity_score_match,
    estimate_ipw,
    estimate_aipw,
    estimate_dml,
)


def generate_synthetic_dataset(
    n: int = 1000,
    true_ate: float = 2.0,
    confounding_strength: float = 1.0,
    non_linear: bool = False,
    unmeasured_confounding: bool = False,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic observational data with a known ground-truth causal effect (tau).
    
    Parameters:
        n: Number of observations
        true_ate: Ground truth average treatment effect
        confounding_strength: Multiplier on confounding effects
        non_linear: If True, introduces non-linear confounding (misspecification test)
        unmeasured_confounding: If True, simulates an unobserved common cause U
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Observed pre-treatment covariates
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(30, 8, n).clip(18, 65)  # e.g., age
    x3 = rng.binomial(1, 0.45, n)            # e.g., binary indicator
    x4 = rng.exponential(1.5, n)             # e.g., prior visit rate

    # Unmeasured confounder U (optional)
    if unmeasured_confounding:
        u = rng.normal(0, 1.5, n)
    else:
        u = np.zeros(n)

    # Treatment Propensity Model logit(P(T=1|X))
    if non_linear:
        # Complex non-linear confounding
        logit_t = (
            -0.8
            + confounding_strength * (0.6 * x1 + 0.03 * (x2 - 30) - 0.4 * x3 + 0.25 * np.log1p(x4)
                                      + 0.15 * (x1 ** 2) - 0.2 * np.sin(x1 * x3))
            + (1.2 * u if unmeasured_confounding else 0.0)
        )
    else:
        # Linear confounding
        logit_t = (
            -0.6
            + confounding_strength * (0.5 * x1 + 0.04 * (x2 - 30) - 0.35 * x3 + 0.2 * x4)
            + (1.0 * u if unmeasured_confounding else 0.0)
        )

    prob_t = 1.0 / (1.0 + np.exp(-logit_t))
    prob_t = np.clip(prob_t, 0.02, 0.98)
    treatment = rng.binomial(1, prob_t)

    # Outcome Model
    if non_linear:
        baseline_y = (
            5.0
            + confounding_strength * (1.2 * x1 + 0.08 * x2 - 1.5 * x3 + 0.7 * np.sqrt(x4)
                                      + 0.3 * (x1 ** 2) + 0.4 * x1 * x3)
            + (1.5 * u if unmeasured_confounding else 0.0)
        )
    else:
        baseline_y = (
            4.0
            + confounding_strength * (1.0 * x1 + 0.07 * x2 - 1.2 * x3 + 0.5 * x4)
            + (1.2 * u if unmeasured_confounding else 0.0)
        )

    noise = rng.normal(0, 1.2, n)
    outcome = true_ate * treatment + baseline_y + noise

    df_dict = {
        "x1_feature": x1,
        "age": x2,
        "binary_flag": x3,
        "prior_activity": x4,
        "treatment": treatment,
        "outcome": outcome,
    }
    if unmeasured_confounding:
        df_dict["_latent_unmeasured_u"] = u

    df = pd.DataFrame(df_dict)
    metadata = {
        "true_ate": float(true_ate),
        "n": int(n),
        "confounding_strength": float(confounding_strength),
        "non_linear": bool(non_linear),
        "unmeasured_confounding": bool(unmeasured_confounding),
        "treatment_prevalence": float(np.mean(treatment)),
        "seed": int(seed),
        "observed_covariates": ["x1_feature", "age", "binary_flag", "prior_activity"],
    }
    return df, metadata


def run_monte_carlo_benchmark(
    n_replications: int = 30,
    n_samples: int = 500,
    true_ate: float = 2.0,
    confounding_strength: float = 1.0,
    non_linear: bool = False,
    unmeasured_confounding: bool = False,
    estimators: List[str] = None,
    seed_start: int = 100
) -> Dict[str, Any]:
    """
    Run Monte Carlo replications to evaluate estimator validity.
    Computes empirical Bias, Variance, RMSE, and 95% Confidence Interval Coverage.
    """
    if estimators is None:
        estimators = ["OLS", "Matching", "IPW", "AIPW", "DML"]

    # Storage for estimates, SEs, and coverage flags
    history: Dict[str, Dict[str, List[float]]] = {
        est: {"estimates": [], "se": [], "covered": []} for est in estimators
    }

    covariates = ["x1_feature", "age", "binary_flag", "prior_activity"]

    for rep in range(n_replications):
        rep_seed = seed_start + rep
        df, _ = generate_synthetic_dataset(
            n=n_samples,
            true_ate=true_ate,
            confounding_strength=confounding_strength,
            non_linear=non_linear,
            unmeasured_confounding=unmeasured_confounding,
            seed=rep_seed,
        )

        for est in estimators:
            try:
                if est == "OLS":
                    res = estimate_ate(df, "treatment", "outcome", covariates)
                elif est == "Matching":
                    res = propensity_score_match(df, "treatment", "outcome", covariates, caliper_sd=0.25)
                elif est == "IPW":
                    res = estimate_ipw(df, "treatment", "outcome", covariates)
                elif est == "AIPW":
                    res = estimate_aipw(df, "treatment", "outcome", covariates)
                elif est == "DML":
                    res = estimate_dml(df, "treatment", "outcome", covariates, n_splits=5, random_state=rep_seed)
                else:
                    continue

                ate_est = res["ate"]
                se_est = res.get("se", 0.0)
                ci_l = res["ci_low"]
                ci_h = res["ci_high"]

                history[est]["estimates"].append(float(ate_est))
                history[est]["se"].append(float(se_est))
                is_cov = (ci_l <= true_ate <= ci_h)
                history[est]["covered"].append(1.0 if is_cov else 0.0)
            except Exception:
                continue

    summary_rows = []
    for est in estimators:
        ests = np.array(history[est]["estimates"])
        if len(ests) == 0:
            continue
        mean_est = float(np.mean(ests))
        bias = float(mean_est - true_ate)
        rel_bias_pct = float(bias / true_ate * 100) if abs(true_ate) > 1e-6 else 0.0
        var_est = float(np.var(ests, ddof=1)) if len(ests) > 1 else 0.0
        sd_est = float(np.sqrt(var_est))
        rmse = float(np.sqrt(bias ** 2 + var_est))
        coverage_rate = float(np.mean(history[est]["covered"]) * 100) if history[est]["covered"] else 0.0
        mean_se = float(np.mean(history[est]["se"])) if history[est]["se"] else 0.0

        summary_rows.append({
            "estimator": est,
            "replications": len(ests),
            "mean_estimate": round(mean_est, 4),
            "bias": round(bias, 4),
            "relative_bias_pct": round(rel_bias_pct, 2),
            "std_error_empirical": round(sd_est, 4),
            "mean_se_reported": round(mean_se, 4),
            "rmse": round(rmse, 4),
            "coverage_95_pct": round(coverage_rate, 1),
        })

    summary_df = pd.DataFrame(summary_rows)

    return {
        "metadata": {
            "true_ate": true_ate,
            "n_replications_planned": n_replications,
            "n_samples": n_samples,
            "confounding_strength": confounding_strength,
            "non_linear": non_linear,
            "unmeasured_confounding": unmeasured_confounding,
        },
        "summary_table": summary_df,
        "replication_data": {
            est: history[est]["estimates"] for est in estimators
        }
    }

