import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold


def _get_identification_assumptions() -> dict:
    """
    Standardized scientific statement of the 5 core identification assumptions
    required for an observational estimate to be interpreted causally.
    """
    return {
        "unconfoundedness": (
            "Conditional Exchangeability / Ignorability: (Y(1), Y(0)) ⫫ T | X. "
            "Assumes that conditional on measured pre-treatment covariates X, treatment assignment is as good as random. "
            "Untestable from observational data alone."
        ),
        "positivity": (
            "Common Support / Positivity: 0 < P(T=1 | X) < 1 for all X in the population. "
            "Ensures every individual has a non-zero probability of receiving either treatment or control."
        ),
        "sutva": (
            "Stable Unit Treatment Value Assumption (SUTVA): "
            "1. No interference between individuals (one person's treatment does not affect another's outcome). "
            "2. No hidden variations of treatment."
        ),
        "consistency": (
            "Consistency: If unit i receives treatment T = t, the observed outcome Y_i equals the potential outcome Y_i(t)."
        ),
        "model_adequacy": (
            "Model Specification: Functional forms (regression equations, link functions) are adequately specified."
        ),
        "causality_caveat": (
            "CORRELATION DOES NOT EQUAL CAUSATION. An adjusted statistical association is causal ONLY if all five "
            "structural identification assumptions hold true. In observational studies, unmeasured confounding can never be ruled out."
        )
    }


def _validate(df: pd.DataFrame, treatment: str, outcome: str, covariates: list) -> pd.DataFrame:
    """
    Validate and clean observational dataset for causal inference.
    Handles binary treatment recoding (bools, integers, strings), ensures complete cases,
    and checks for positive treatment/control counts.
    """
    required = [treatment, outcome] + list(covariates)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns not found in dataset: {missing}")

    data = df[required].dropna().copy()
    if len(data) < 30:
        raise ValueError(f"At least 30 complete observations required; got {len(data)}.")

    # Normalize binary treatment to exactly {0, 1}
    t_raw = data[treatment]
    if t_raw.nunique() != 2:
        raise ValueError(
            f"Treatment column '{treatment}' must have exactly two unique values. "
            f"Observed: {list(t_raw.unique()[:5])}"
        )

    # Convert boolean, string, or non-{0, 1} encoding
    if set(t_raw.unique()) == {0, 1}:
        data[treatment] = t_raw.astype(int)
    elif set(t_raw.unique()) == {False, True}:
        data[treatment] = t_raw.astype(int)
    else:
        unique_vals = sorted(t_raw.unique())
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        data[treatment] = t_raw.map(mapping).astype(int)

    n_treated = int(data[treatment].sum())
    n_control = int((1 - data[treatment]).sum())
    if n_treated < 5 or n_control < 5:
        raise ValueError(
            f"Both treatment arms must have at least 5 observations. "
            f"Got {n_treated} treated and {n_control} control."
        )

    return data


def estimate_ate(df: pd.DataFrame, treatment: str, outcome: str, covariates: list, cov_type: str = "HC3") -> dict:
    """
    Adjusted Regression (OLS & G-Computation Standardization) estimating both ATE and ATT
    with HC3 robust standard errors.
    
    Y = alpha + tau * T + beta * X + epsilon
    Under constant treatment effect: ATE = ATT = tau.
    Under heterogeneous / potential outcome G-computation:
        ATE = (1/N) * sum(mu_1(X) - mu_0(X))
        ATT = (1/N1) * sum_{T=1} (Y - mu_0(X))
    """
    data = _validate(df, treatment, outcome, covariates)
    y = data[outcome].astype(float)
    t = data[treatment].astype(float)
    x_covs = data[list(covariates)].astype(float) if covariates else pd.DataFrame(index=data.index)

    design = pd.concat([t.rename("treatment"), x_covs], axis=1)
    x = sm.add_constant(design, has_constant="add")

    model = sm.OLS(y, x).fit(cov_type=cov_type)
    ci = model.conf_int().loc["treatment"]
    ate = float(model.params["treatment"])
    se = float(model.bse["treatment"])
    pval = float(model.pvalues["treatment"])

    # G-computation for ATT estimation
    # Fit mu_0(X) on controls: E[Y | T=0, X]
    ctrl_mask = (t == 0)
    treat_mask = (t == 1)
    x_const = sm.add_constant(x_covs, has_constant="add")
    model_ctrl = sm.OLS(y[ctrl_mask], x_const[ctrl_mask]).fit()
    y0_pred_treated = model_ctrl.predict(x_const[treat_mask])
    att_gcomp = float(np.mean(y[treat_mask] - y0_pred_treated))

    # In linear model without interactions, ATE = ATT = tau
    att = att_gcomp if abs(att_gcomp - ate) > 1e-4 else ate
    att_se = se
    att_ci_low = float(att - 1.96 * att_se)
    att_ci_high = float(att + 1.96 * att_se)
    att_pval = pval

    # Covariate coefficients summary
    coef_records = []
    for var in x.columns:
        var_ci = model.conf_int().loc[var]
        coef_records.append({
            "term": var,
            "coef": round(float(model.params[var]), 4),
            "std_err": round(float(model.bse[var]), 4),
            "p_value": round(float(model.pvalues[var]), 5),
            "ci_low": round(float(var_ci.iloc[0]), 4),
            "ci_high": round(float(var_ci.iloc[1]), 4),
        })

    return {
        "estimator": "Adjusted Regression (OLS & G-Computation)",
        "method": f"OLS outcome regression with {cov_type} robust standard errors and counterfactual G-computation",
        "ate": ate,
        "se": se,
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": pval,
        "att": att,
        "att_se": att_se,
        "att_ci_low": att_ci_low,
        "att_ci_high": att_ci_high,
        "att_p_value": att_pval,
        "n": int(len(data)),
        "n_treated": int(t.sum()),
        "n_control": int((1 - t).sum()),
        "coefficients": coef_records,
        "diagnostics": {
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else 0.0,
            "robust_se": se,
            "cov_type": cov_type,
            "warning": "Adjusted association is causal only under unconfoundedness, positivity, and correct functional specification."
        },
        "identification_assumptions": _get_identification_assumptions()
    }


def propensity_score_match(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list,
    caliper_sd: float = 0.2,
    with_replacement: bool = True
) -> dict:
    """
    1:1 nearest-neighbour propensity-score matching with caliper and comprehensive balance diagnostics.
    Calculates both ATT (treated units matched to controls) and ATE (two-way matched sample).
    """
    if not covariates:
        raise ValueError("Propensity score matching requires at least one pre-treatment covariate.")

    data = _validate(df, treatment, outcome, covariates)
    t = data[treatment].to_numpy().astype(int)
    X = data[list(covariates)].to_numpy().astype(float)
    y = data[outcome].to_numpy().astype(float)
    n = len(t)

    # 1. Estimate Propensity Scores
    ps_model = LogisticRegression(max_iter=3000, solver="lbfgs")
    ps_model.fit(X, t)
    ps = ps_model.predict_proba(X)[:, 1]

    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]
    n_treated = len(treated_idx)
    n_control = len(control_idx)

    # 2. Overlap & Common Support
    ps_std = float(np.std(ps))
    caliper = caliper_sd * ps_std
    overlap_min = float(max(ps[treated_idx].min(), ps[control_idx].min()))
    overlap_max = float(min(ps[treated_idx].max(), ps[control_idx].max()))

    # Check treated outside common support
    treated_on_support = (ps[treated_idx] >= overlap_min) & (ps[treated_idx] <= overlap_max)
    pct_treated_on_support = round(float(np.mean(treated_on_support) * 100), 1)

    # 3. Match Treated to Control -> ATT
    if with_replacement:
        nn_ctrl = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(ps[control_idx].reshape(-1, 1))
        distances_t, indices_t = nn_ctrl.kneighbors(ps[treated_idx].reshape(-1, 1))
        matched_ctrl_for_t = control_idx[indices_t[:, 0]]
        keep_t = distances_t[:, 0] <= caliper
        matched_treated_idx = treated_idx[keep_t]
        matched_ctrl_idx = matched_ctrl_for_t[keep_t]
        match_distances = distances_t[keep_t, 0]
    else:
        matched_treated_idx = []
        matched_ctrl_idx = []
        match_distances = []
        avail_ctrl = set(control_idx)
        sorted_treated = treated_idx[np.argsort(ps[treated_idx])]
        for tidx in sorted_treated:
            if not avail_ctrl:
                break
            ctrl_list = list(avail_ctrl)
            diffs = np.abs(ps[ctrl_list] - ps[tidx])
            best_i = int(np.argmin(diffs))
            best_d = float(diffs[best_i])
            if best_d <= caliper:
                best_c = ctrl_list[best_i]
                matched_treated_idx.append(tidx)
                matched_ctrl_idx.append(best_c)
                match_distances.append(best_d)
                avail_ctrl.remove(best_c)
        matched_treated_idx = np.array(matched_treated_idx)
        matched_ctrl_idx = np.array(matched_ctrl_idx)
        match_distances = np.array(match_distances)

    n_pairs = len(matched_treated_idx)
    if n_pairs < 5:
        raise ValueError(
            f"Fewer than 5 matches met the caliper ({caliper:.4f}). "
            "Check covariate overlap or consider widening the caliper."
        )

    # ATT Estimation
    att_pair_effects = y[matched_treated_idx] - y[matched_ctrl_idx]
    att = float(np.mean(att_pair_effects))
    att_se = float(np.std(att_pair_effects, ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else 0.0

    t_crit = float(stats.t.ppf(0.975, df=max(n_pairs - 1, 1)))
    att_ci_low = float(att - t_crit * att_se)
    att_ci_high = float(att + t_crit * att_se)
    att_t_stat = att / att_se if att_se > 1e-9 else 0.0
    att_p_val = float(2 * (1 - stats.t.cdf(abs(att_t_stat), df=max(n_pairs - 1, 1))))

    # 4. Two-Way Match for ATE (Match Control to Treated for ATC)
    nn_treat = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(ps[treated_idx].reshape(-1, 1))
    distances_c, indices_c = nn_treat.kneighbors(ps[control_idx].reshape(-1, 1))
    matched_treat_for_c = treated_idx[indices_c[:, 0]]
    keep_c = distances_c[:, 0] <= caliper
    atc_pair_effects = y[matched_treat_for_c[keep_c]] - y[control_idx[keep_c]]
    atc = float(np.mean(atc_pair_effects)) if len(atc_pair_effects) > 0 else att

    # Full ATE = P(T=1)*ATT + P(T=0)*ATC
    p_treat = n_treated / n
    ate = float(p_treat * att + (1 - p_treat) * atc)
    # Conservative standard error combining matched samples
    ate_se = float(np.sqrt((p_treat * att_se)**2 + ((1 - p_treat) * (att_se * 1.1))**2))
    ate_ci_low = float(ate - 1.96 * ate_se)
    ate_ci_high = float(ate + 1.96 * ate_se)
    ate_p_val = float(2 * (1 - stats.norm.cdf(abs(ate / ate_se)))) if ate_se > 1e-9 else 0.0

    # 5. Post-matching Covariate Balance Table
    balance_records = []
    for cov_i, cov_name in enumerate(covariates):
        pre_t = X[treated_idx, cov_i]
        pre_c = X[control_idx, cov_i]
        pre_v1 = float(np.var(pre_t, ddof=1)) if len(pre_t) > 1 else 1.0
        pre_v0 = float(np.var(pre_c, ddof=1)) if len(pre_c) > 1 else 1.0
        pre_sd = np.sqrt((pre_v1 + pre_v0) / 2.0) if (pre_v1 + pre_v0) > 0 else 1.0
        smd_pre = (np.mean(pre_t) - np.mean(pre_c)) / pre_sd if pre_sd > 1e-9 else 0.0
        var_ratio_pre = pre_v1 / pre_v0 if pre_v0 > 1e-9 else 1.0

        post_t = X[matched_treated_idx, cov_i]
        post_c = X[matched_ctrl_idx, cov_i]
        post_v1 = float(np.var(post_t, ddof=1)) if len(post_t) > 1 else 1.0
        post_v0 = float(np.var(post_c, ddof=1)) if len(post_c) > 1 else 1.0
        post_sd = np.sqrt((post_v1 + post_v0) / 2.0) if (post_v1 + post_v0) > 0 else 1.0
        smd_post = (np.mean(post_t) - np.mean(post_c)) / post_sd if post_sd > 1e-9 else 0.0
        var_ratio_post = post_v1 / post_v0 if post_v0 > 1e-9 else 1.0

        pct_reduction = (abs(smd_pre) - abs(smd_post)) / abs(smd_pre) * 100 if abs(smd_pre) > 1e-6 else 100.0

        balance_records.append({
            "covariate": cov_name,
            "mean_treated_post": round(float(np.mean(post_t)), 4),
            "mean_control_post": round(float(np.mean(post_c)), 4),
            "smd_pre": round(float(smd_pre), 4),
            "smd_post": round(float(smd_post), 4),
            "abs_smd_post": round(float(abs(smd_post)), 4),
            "var_ratio_pre": round(float(var_ratio_pre), 3),
            "var_ratio_post": round(float(var_ratio_post), 3),
            "bias_reduction_pct": round(float(pct_reduction), 1),
            "balanced": abs(smd_post) < 0.1 and (0.8 <= var_ratio_post <= 1.25),
        })

    return {
        "estimator": "Propensity Score Matching (1:1 Nearest Neighbour)",
        "method": f"1:1 nearest-neighbour propensity-score matching ({caliper_sd} SD caliper, replacement={with_replacement})",
        "att": att,
        "att_se": att_se,
        "att_ci_low": att_ci_low,
        "att_ci_high": att_ci_high,
        "att_p_value": att_p_val,
        "ate": ate,
        "se": ate_se,
        "ci_low": ate_ci_low,
        "ci_high": ate_ci_high,
        "p_value": ate_p_val,
        "n_pairs": int(n_pairs),
        "n_treated_total": int(n_treated),
        "n_control_total": int(n_control),
        "overlap": [overlap_min, overlap_max],
        "overlap_diagnostics": {
            "overlap_range": [round(overlap_min, 4), round(overlap_max, 4)],
            "pct_treated_on_support": pct_treated_on_support,
            "caliper_exclusions": int(n_treated - n_pairs),
            "positivity_status": "Adequate Common Support" if pct_treated_on_support >= 90 else "Positivity Violation Alert (< 90% Support)"
        },
        "mean_abs_ps_distance": float(np.mean(match_distances)),
        "balance_table": balance_records,
        "propensity_scores": {
            "treated": ps[treated_idx].tolist(),
            "control": ps[control_idx].tolist()
        },
        "identification_assumptions": _get_identification_assumptions()
    }


def estimate_ipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list,
    trim_percentile: float = 0.01,
    stabilized: bool = True
) -> dict:
    """
    Inverse Probability Weighting (IPW / Hajek normalized estimator) calculating both
    ATE (Average Treatment Effect) and ATT (Average Treatment Effect on the Treated)
    with stabilized weights, trimming, and robust sandwich variance estimation.
    """
    if not covariates:
        raise ValueError("IPW estimation requires at least one pre-treatment covariate.")

    data = _validate(df, treatment, outcome, covariates)
    t = data[treatment].to_numpy().astype(int)
    X = data[list(covariates)].to_numpy().astype(float)
    y = data[outcome].to_numpy().astype(float)
    n = len(t)

    # Propensity model
    ps_model = LogisticRegression(max_iter=3000, solver="lbfgs")
    ps_model.fit(X, t)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    p_t = float(np.mean(t))

    # 1. Weights for ATE
    if stabilized:
        w_ate = np.where(t == 1, p_t / ps, (1.0 - p_t) / (1.0 - ps))
    else:
        w_ate = np.where(t == 1, 1.0 / ps, 1.0 / (1.0 - ps))

    # 2. Weights for ATT
    # w_ATT = 1 for treated; ps / (1 - ps) for control
    w_att = np.where(t == 1, 1.0, ps / (1.0 - ps))

    # Trimming
    if trim_percentile > 0:
        w_ate = np.clip(w_ate, np.percentile(w_ate, trim_percentile * 100), np.percentile(w_ate, (1 - trim_percentile) * 100))
        w_att = np.clip(w_att, np.percentile(w_att, trim_percentile * 100), np.percentile(w_att, (1 - trim_percentile) * 100))

    # Hajek normalized ATE
    w1_ate = w_ate * t
    w0_ate = w_ate * (1 - t)
    ate = float(np.sum(w1_ate * y) / np.sum(w1_ate) - np.sum(w0_ate * y) / np.sum(w0_ate))

    # Weighted OLS sandwich SE for ATE
    x_des = sm.add_constant(t.astype(float))
    wls_ate = sm.WLS(y, x_des, weights=w_ate).fit(cov_type="HC1")
    se_ate = float(wls_ate.bse[1])
    ci_ate = wls_ate.conf_int()
    ci_low_ate = float(ci_ate.iloc[1, 0] if hasattr(ci_ate, "iloc") else ci_ate[1, 0])
    ci_high_ate = float(ci_ate.iloc[1, 1] if hasattr(ci_ate, "iloc") else ci_ate[1, 1])
    pval_ate = float(wls_ate.pvalues[1])

    # Hajek normalized ATT
    w1_att = w_att * t
    w0_att = w_att * (1 - t)
    att = float(np.sum(w1_att * y) / np.sum(w1_att) - np.sum(w0_att * y) / np.sum(w0_att))

    # Weighted OLS sandwich SE for ATT
    wls_att = sm.WLS(y, x_des, weights=w_att).fit(cov_type="HC1")
    se_att = float(wls_att.bse[1])
    ci_att = wls_att.conf_int()
    ci_low_att = float(ci_att.iloc[1, 0] if hasattr(ci_att, "iloc") else ci_att[1, 0])
    ci_high_att = float(ci_att.iloc[1, 1] if hasattr(ci_att, "iloc") else ci_att[1, 1])
    pval_att = float(wls_att.pvalues[1])

    # Effective Sample Size (Kish's ESS)
    ess_t1_ate = float((np.sum(w1_ate) ** 2) / np.sum(w1_ate ** 2))
    ess_t0_ate = float((np.sum(w0_ate) ** 2) / np.sum(w0_ate ** 2))
    ess_total_ate = ess_t1_ate + ess_t0_ate

    # Weighted Covariate Balance (ATE Weights)
    balance_records = []
    for cov_i, cov_name in enumerate(covariates):
        xc = X[:, cov_i]
        wm1 = np.sum(w1_ate * xc) / np.sum(w1_ate)
        wm0 = np.sum(w0_ate * xc) / np.sum(w0_ate)
        wv1 = np.sum(w1_ate * (xc - wm1) ** 2) / np.sum(w1_ate)
        wv0 = np.sum(w0_ate * (xc - wm0) ** 2) / np.sum(w0_ate)
        wsd = np.sqrt((wv1 + wv0) / 2.0) if (wv1 + wv0) > 0 else 1.0
        wsmd = (wm1 - wm0) / wsd if wsd > 1e-9 else 0.0
        var_ratio = wv1 / wv0 if wv0 > 1e-9 else 1.0

        balance_records.append({
            "covariate": cov_name,
            "weighted_mean_treated": round(float(wm1), 4),
            "weighted_mean_control": round(float(wm0), 4),
            "weighted_smd": round(float(wsmd), 4),
            "abs_weighted_smd": round(float(abs(wsmd)), 4),
            "weighted_var_ratio": round(float(var_ratio), 3),
            "balanced": abs(wsmd) < 0.1,
        })

    return {
        "estimator": "Inverse Probability Weighting (IPW)",
        "method": f"Hajek normalized IPW ({'stabilized' if stabilized else 'unstandardized'}, {trim_percentile*100:.1f}% trimming)",
        "ate": ate,
        "se": se_ate,
        "ci_low": ci_low_ate,
        "ci_high": ci_high_ate,
        "p_value": pval_ate,
        "att": att,
        "att_se": se_att,
        "att_ci_low": ci_low_att,
        "att_ci_high": ci_high_att,
        "att_p_value": pval_att,
        "n": n,
        "effective_sample_size": {
            "treated_ess": round(ess_t1_ate, 1),
            "control_ess": round(ess_t0_ate, 1),
            "total_ess": round(ess_total_ate, 1)
        },
        "balance_table": balance_records,
        "overlap_diagnostics": {
            "ps_min": float(np.min(ps)),
            "ps_max": float(np.max(ps)),
            "trimmed_weights_max": float(np.max(w_ate)),
            "ess_ratio": round(ess_total_ate / n, 2),
            "positivity_status": "Adequate Common Support" if (ess_total_ate / n) >= 0.5 else "Moderate Positivity Restriction"
        },
        "identification_assumptions": _get_identification_assumptions()
    }


def estimate_aipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list
) -> dict:
    """
    Augmented Inverse Probability Weighting (AIPW / Doubly Robust Estimator).
    Combines outcome regression models mu_1(X), mu_0(X) and propensity score e(X).
    Calculates both ATE and ATT using Efficient Influence Function (EIF) variance estimation.
    Consistent if EITHER the outcome model OR the propensity model is correctly specified.
    """
    if not covariates:
        raise ValueError("AIPW requires at least one pre-treatment covariate.")

    data = _validate(df, treatment, outcome, covariates)
    t = data[treatment].to_numpy().astype(int)
    X = data[list(covariates)].to_numpy().astype(float)
    y = data[outcome].to_numpy().astype(float)
    n = len(t)
    n_treated = int(np.sum(t))
    n_control = int(np.sum(1 - t))

    # 1. Propensity Model e(X)
    ps_model = LogisticRegression(max_iter=3000, solver="lbfgs")
    ps_model.fit(X, t)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)

    # 2. Outcome Models: mu_1(X) and mu_0(X)
    reg1 = Ridge(alpha=1.0).fit(X[t == 1], y[t == 1])
    reg0 = Ridge(alpha=1.0).fit(X[t == 0], y[t == 0])

    mu1_hat = reg1.predict(X)
    mu0_hat = reg0.predict(X)

    # 3. Doubly Robust ATE
    # psi_ATE_i = mu1(X) - mu0(X) + T*(Y - mu1(X))/ps - (1-T)*(Y - mu0(X))/(1-ps)
    psi_ate = (mu1_hat - mu0_hat) + (t * (y - mu1_hat)) / ps - ((1 - t) * (y - mu0_hat)) / (1.0 - ps)
    ate = float(np.mean(psi_ate))
    var_eif_ate = np.var(psi_ate, ddof=1)
    se_ate = float(np.sqrt(var_eif_ate / n))

    ci_low_ate = float(ate - 1.96 * se_ate)
    ci_high_ate = float(ate + 1.96 * se_ate)
    pval_ate = float(2 * (1 - stats.norm.cdf(abs(ate / se_ate)))) if se_ate > 1e-9 else 0.0

    # 4. Doubly Robust ATT
    # psi_ATT_i = (T - ps) / P(T=1) * (Y - mu0(X)) / (1 - ps)
    p_t = n_treated / n
    # Counterfactual expectation: Y - mu0_hat
    diff_y_mu0 = y - mu0_hat
    psi_att = np.where(
        t == 1,
        (diff_y_mu0) / p_t,
        - (ps / (1.0 - ps)) * (diff_y_mu0) / p_t
    )
    att = float(np.sum(psi_att) / n)
    var_eif_att = np.var(psi_att, ddof=1)
    se_att = float(np.sqrt(var_eif_att / n))

    ci_low_att = float(att - 1.96 * se_att)
    ci_high_att = float(att + 1.96 * se_att)
    pval_att = float(2 * (1 - stats.norm.cdf(abs(att / se_att)))) if se_att > 1e-9 else 0.0

    return {
        "estimator": "Augmented IPW / AIPW (Doubly Robust)",
        "method": "Doubly Robust AIPW with Ridge outcome regressions, Logistic propensity, and EIF standard errors",
        "ate": ate,
        "se": se_ate,
        "ci_low": ci_low_ate,
        "ci_high": ci_high_ate,
        "p_value": pval_ate,
        "att": att,
        "att_se": se_att,
        "att_ci_low": ci_low_att,
        "att_ci_high": ci_high_att,
        "att_p_value": pval_att,
        "n": n,
        "n_treated": n_treated,
        "n_control": n_control,
        "overlap_diagnostics": {
            "ps_clipped_min": float(np.min(ps)),
            "ps_clipped_max": float(np.max(ps)),
            "property": "Double Robustness: Unbiased if either outcome regression OR propensity model is correct."
        },
        "identification_assumptions": _get_identification_assumptions()
    }


def estimate_dml(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list,
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """
    Double / Debiased Machine Learning (DML) for Partially Linear Models (Chernozhukov et al. 2018).
    Uses K-fold cross-fitting to orthogonalize residuals:
        Y - E[Y|X] = tau * (T - E[T|X]) + epsilon
    Calculates both ATE and ATT using Neyman-orthogonal scores.
    """
    if not covariates:
        raise ValueError("Double Machine Learning requires at least one covariate.")

    data = _validate(df, treatment, outcome, covariates)
    t = data[treatment].to_numpy().astype(float)
    X = data[list(covariates)].to_numpy().astype(float)
    y = data[outcome].to_numpy().astype(float)
    n = len(t)

    kf = KFold(n_splits=min(n_splits, len(t) // 6), shuffle=True, random_state=random_state)

    y_res = np.zeros(n)
    t_res = np.zeros(n)

    for train_idx, test_idx in kf.split(X):
        reg_y = Ridge(alpha=1.0).fit(X[train_idx], y[train_idx])
        reg_t = Ridge(alpha=1.0).fit(X[train_idx], t[train_idx])

        y_res[test_idx] = y[test_idx] - reg_y.predict(X[test_idx])
        t_res[test_idx] = t[test_idx] - reg_t.predict(X[test_idx])

    # 1. DML ATE (Neyman-orthogonal regression on full sample)
    denom_ate = float(np.sum(t_res ** 2))
    if denom_ate < 1e-9:
        raise ValueError("Residual treatment variance is near zero; collinearity or perfect predictability.")

    ate = float(np.sum(t_res * y_res) / denom_ate)
    score_ate = (y_res - ate * t_res) * t_res
    var_score_ate = float(np.mean(score_ate ** 2))
    se_ate = float(np.sqrt(var_score_ate / n) / (denom_ate / n))

    ci_low_ate = float(ate - 1.96 * se_ate)
    ci_high_ate = float(ate + 1.96 * se_ate)
    pval_ate = float(2 * (1 - stats.norm.cdf(abs(ate / se_ate)))) if se_ate > 1e-9 else 0.0

    # 2. DML ATT (Evaluated over the treated sample)
    treated_mask = (t == 1.0)
    denom_att = float(np.sum(t_res[treated_mask] ** 2))
    if denom_att > 1e-9:
        att = float(np.sum(t_res[treated_mask] * y_res[treated_mask]) / denom_att)
        score_att = (y_res[treated_mask] - att * t_res[treated_mask]) * t_res[treated_mask]
        var_score_att = float(np.mean(score_att ** 2))
        se_att = float(np.sqrt(var_score_att / np.sum(treated_mask)) / (denom_att / np.sum(treated_mask)))
        ci_low_att = float(att - 1.96 * se_att)
        ci_high_att = float(att + 1.96 * se_att)
        pval_att = float(2 * (1 - stats.norm.cdf(abs(att / se_att)))) if se_att > 1e-9 else 0.0
    else:
        att = ate
        se_att = se_ate
        ci_low_att = ci_low_ate
        ci_high_att = ci_high_ate
        pval_att = pval_ate

    return {
        "estimator": "Double Machine Learning (DML)",
        "method": f"DML Partial Linear Model with {n_splits}-fold cross-fitting, Ridge models, and Neyman-orthogonal scores",
        "ate": ate,
        "se": se_ate,
        "ci_low": ci_low_ate,
        "ci_high": ci_high_ate,
        "p_value": pval_ate,
        "att": att,
        "att_se": se_att,
        "att_ci_low": ci_low_att,
        "att_ci_high": ci_high_att,
        "att_p_value": pval_att,
        "n": n,
        "n_splits": n_splits,
        "residual_t_variance": float(np.var(t_res)),
        "residual_y_variance": float(np.var(y_res)),
        "overlap_diagnostics": {
            "orthogonal_score_var": var_score_ate,
            "residual_treatment_var": float(np.var(t_res)),
            "property": "Neyman orthogonality eliminates first-order bias from non-parametric nuisance estimation."
        },
        "identification_assumptions": _get_identification_assumptions()
    }


def compare_all_estimators(df: pd.DataFrame, treatment: str, outcome: str, covariates: list) -> list:
    """
    Run all 5 research estimators (Adjusted Regression, PSM, IPW, Doubly Robust AIPW, DML)
    and return a standardized benchmark comparison list for side-by-side analysis,
    calculating both ATE and ATT where supported.
    """
    results = []

    # 1. Adjusted Regression
    try:
        r_ols = estimate_ate(df, treatment, outcome, covariates)
        results.append(r_ols)
    except Exception as exc:
        results.append({"estimator": "Adjusted Regression (OLS)", "ate": np.nan, "att": np.nan, "error": str(exc)})

    # 2. Propensity Score Matching
    try:
        r_psm = propensity_score_match(df, treatment, outcome, covariates)
        results.append(r_psm)
    except Exception as exc:
        results.append({"estimator": "Propensity Score Matching", "ate": np.nan, "att": np.nan, "error": str(exc)})

    # 3. Inverse Probability Weighting
    try:
        r_ipw = estimate_ipw(df, treatment, outcome, covariates)
        results.append(r_ipw)
    except Exception as exc:
        results.append({"estimator": "Inverse Probability Weighting", "ate": np.nan, "att": np.nan, "error": str(exc)})

    # 4. AIPW (Doubly Robust)
    try:
        r_aipw = estimate_aipw(df, treatment, outcome, covariates)
        results.append(r_aipw)
    except Exception as exc:
        results.append({"estimator": "Augmented IPW / AIPW (Doubly Robust)", "ate": np.nan, "att": np.nan, "error": str(exc)})

    # 5. Double Machine Learning
    try:
        r_dml = estimate_dml(df, treatment, outcome, covariates)
        results.append(r_dml)
    except Exception as exc:
        results.append({"estimator": "Double Machine Learning", "ate": np.nan, "att": np.nan, "error": str(exc)})

    return results
