import numpy as np
import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a research-grade data quality and distribution profile for a DataFrame.
    Preserves original columns ('column', 'dtype', 'missing', 'missing_pct',
    'unique', 'sample_values') while enriching with summary statistics.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        clean = s.dropna()
        
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
            "sample_values": ", ".join(map(str, clean.unique()[:4])),
        }
        
        if is_num and len(clean) > 0:
            row["mean"] = round(float(clean.mean()), 3)
            row["std"] = round(float(clean.std()), 3) if len(clean) > 1 else 0.0
            row["min"] = round(float(clean.min()), 3)
            row["median"] = round(float(clean.median()), 3)
            row["max"] = round(float(clean.max()), 3)
            row["zero_count"] = int((clean == 0).sum())
        else:
            row["mean"] = None
            row["std"] = None
            row["min"] = None
            row["median"] = None
            row["max"] = None
            row["zero_count"] = None
            
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations for all numeric features,
    sorted by descending absolute magnitude.
    """
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r"])
    corr = num.corr()
    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            val = corr.loc[a, b]
            if not np.isnan(val):
                pairs.append({
                    "feature_a": a,
                    "feature_b": b,
                    "pearson_r": round(float(val), 4)
                })
    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r"])
    return pd.DataFrame(pairs).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify potential outliers in numerical features using both the
    Tukey Interquartile Range (IQR) and Z-score (|Z| > 3) methods.
    """
    num = df.select_dtypes(include="number")
    records = []
    for col in num.columns:
        s = num[col].dropna()
        if len(s) < 5:
            continue
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        iqr_outliers = int(((s < lower_bound) | (s > upper_bound)).sum())
        
        std = s.std()
        if std > 1e-9:
            z_scores = (s - s.mean()).abs() / std
            z_outliers = int((z_scores > 3.0).sum())
        else:
            z_outliers = 0
            
        records.append({
            "feature": col,
            "iqr_outliers": iqr_outliers,
            "iqr_outlier_pct": round(iqr_outliers / len(s) * 100, 2),
            "z3_outliers": z_outliers,
            "z3_outlier_pct": round(z_outliers / len(s) * 100, 2),
            "lower_bound_iqr": round(float(lower_bound), 3),
            "upper_bound_iqr": round(float(upper_bound), 3),
        })
    return pd.DataFrame(records)


def compute_baseline_smd(df: pd.DataFrame, treatment: str, covariates: list) -> pd.DataFrame:
    """
    Compute pre-treatment covariate balance diagnostics between treated and control groups.
    Computes Standardized Mean Differences (SMD), variance ratios, and Rubin's diagnostic rating.
    
    Formula:
        SMD = (Mean_T1 - Mean_T0) / sqrt((Var_T1 + Var_T0) / 2)
    """
    req = [treatment] + list(covariates)
    data = df[req].dropna().copy()
    
    t = data[treatment]
    t1_mask = (t == 1)
    t0_mask = (t == 0)
    
    n_t1 = int(t1_mask.sum())
    n_t0 = int(t0_mask.sum())
    if n_t1 == 0 or n_t0 == 0:
        raise ValueError("Both treatment and control groups must contain non-zero observations.")
        
    records = []
    for cov in covariates:
        s1 = data.loc[t1_mask, cov].astype(float)
        s0 = data.loc[t0_mask, cov].astype(float)
        
        m1, m0 = float(s1.mean()), float(s0.mean())
        v1, v0 = float(s1.var()), float(s0.var())
        
        # Pooled standard deviation
        pooled_sd = np.sqrt((v1 + v0) / 2.0) if (v1 + v0) > 0 else 1.0
        smd = (m1 - m0) / pooled_sd if pooled_sd > 1e-9 else 0.0
        
        var_ratio = v1 / v0 if v0 > 1e-9 else (1.0 if v1 <= 1e-9 else np.nan)
        
        abs_smd = abs(smd)
        if abs_smd < 0.1:
            rubin_status = "Balanced (< 0.10)"
        elif abs_smd < 0.25:
            rubin_status = "Marginal (0.10 - 0.25)"
        else:
            rubin_status = "Severe Imbalance (> 0.25)"
            
        records.append({
            "covariate": cov,
            "mean_treated": round(m1, 4),
            "mean_control": round(m0, 4),
            "smd": round(float(smd), 4),
            "abs_smd": round(float(abs_smd), 4),
            "var_ratio": round(float(var_ratio), 3) if not np.isnan(var_ratio) else None,
            "balance_status": rubin_status,
        })
        
    return pd.DataFrame(records).sort_values("abs_smd", ascending=False)
