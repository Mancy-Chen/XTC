import os
import re
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, wilcoxon, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

# =========================================================
# SETTINGS
# =========================================================
input_dir = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC"
output_dir = "/data/projects/CSC/code/XTC/07_regression_model/Output/radiomics_longitudinal_stats"
os.makedirs(output_dir, exist_ok=True)

x_path = f"{input_dir}/X_pre_delta_with_id_filteredQC.csv"
y_path = f"{input_dir}/y_with_id_filteredQC.csv"

outcome_col = "vwrec"   # change if needed
alpha = 0.05

# Restrict to these ROIs
target_rois = [
    "Left_Hippocampus",
    "Right_Hippocampus",
    "Left_Thalamus",
    "Right_Thalamus",
]

# Keep only these feature families
target_feature_keywords = [
    "firstorder",
    "shape",
]

# =========================================================
# LOAD DATA
# =========================================================
X_df = pd.read_csv(x_path)
y_df = pd.read_csv(y_path)

print("Loaded X:", X_df.shape)
print("Loaded y:", y_df.shape)

# Merge y for optional downstream correlation
if outcome_col in y_df.columns:
    merged_y = y_df[["cohort", "id", outcome_col]].copy()
    merged_y[outcome_col] = pd.to_numeric(merged_y[outcome_col], errors="coerce")
else:
    merged_y = None
    print(f"Warning: {outcome_col} not found in y file. Outcome correlation step will be skipped.")

# =========================================================
# FIND PRE/DELTA FEATURE PAIRS AND RECONSTRUCT POST
# =========================================================
all_cols = X_df.columns.tolist()

pre_cols = [c for c in all_cols if c.endswith("_pre")]
delta_cols = [c for c in all_cols if c.endswith("_delta")]

pre_bases = {c[:-4] for c in pre_cols}
delta_bases = {c[:-6] for c in delta_cols}
common_bases = sorted(pre_bases & delta_bases)

def keep_feature(base_name: str) -> bool:
    roi_ok = any(base_name.startswith(roi + "_") for roi in target_rois)
    feat_ok = any(k in base_name for k in target_feature_keywords)
    return roi_ok and feat_ok

feature_bases = [b for b in common_bases if keep_feature(b)]

print("Number of paired ROI feature bases:", len(feature_bases))

# convert numeric
for col in [f"{b}_pre" for b in feature_bases] + [f"{b}_delta" for b in feature_bases]:
    if X_df[col].dtype == object:
        X_df[col] = X_df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

# reconstruct post
for base in feature_bases:
    X_df[f"{base}_post"] = X_df[f"{base}_pre"] + X_df[f"{base}_delta"]

# =========================================================
# CONVERT NUMERIC
# =========================================================
for col in [f"{b}_pre" for b in feature_bases] + [f"{b}_post" for b in feature_bases]:
    if X_df[col].dtype == object:
        X_df[col] = X_df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

# =========================================================
# HELPERS
# =========================================================
def parse_roi_and_feature(base_name: str):
    for roi in target_rois:
        prefix = roi + "_"
        if base_name.startswith(prefix):
            return roi, base_name[len(prefix):]
    return None, base_name

def cohens_d_paired(pre, post):
    diff = post - pre
    sd = np.std(diff, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return np.mean(diff) / sd

def rank_biserial_from_wilcoxon(pre, post):
    # Approximation from signed ranks using scipy wilcoxon statistic
    diff = post - pre
    diff = diff[~np.isnan(diff)]
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return np.nan
    try:
        stat = wilcoxon(post, pre, zero_method="wilcox", alternative="two-sided").statistic
        # For two-sided Wilcoxon, statistic is the smaller rank sum.
        # Approximate rank-biserial:
        total_rank_sum = n * (n + 1) / 2
        rbc = 1 - (2 * stat / total_rank_sum)
        return rbc
    except Exception:
        return np.nan

# =========================================================
# PAIRED TESTS
# =========================================================
results = []

for base in feature_bases:
    pre_col = f"{base}_pre"
    post_col = f"{base}_post"

    sub = X_df[[pre_col, post_col]].copy().dropna()
    n = len(sub)

    if n < 3:
        continue

    pre = sub[pre_col].to_numpy(dtype=float)
    post = sub[post_col].to_numpy(dtype=float)
    delta = post - pre

    roi, short_feature = parse_roi_and_feature(base)

    mean_pre = np.mean(pre)
    mean_post = np.mean(post)
    mean_delta = np.mean(delta)
    sd_delta = np.std(delta, ddof=1) if n > 1 else np.nan
    median_delta = np.median(delta)

    # Paired t-test
    try:
        t_stat, p_t = ttest_rel(post, pre, nan_policy="omit")
    except Exception:
        t_stat, p_t = np.nan, np.nan

    # Wilcoxon
    try:
        w_stat, p_w = wilcoxon(post, pre, zero_method="wilcox", alternative="two-sided")
    except Exception:
        w_stat, p_w = np.nan, np.nan

    d_paired = cohens_d_paired(pre, post)
    rbc = rank_biserial_from_wilcoxon(pre, post)

    results.append({
        "roi": roi,
        "feature_base": base,
        "feature_name": short_feature,
        "n": n,
        "mean_pre": mean_pre,
        "mean_post": mean_post,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "sd_delta": sd_delta,
        "t_stat": t_stat,
        "p_ttest": p_t,
        "w_stat": w_stat,
        "p_wilcoxon": p_w,
        "cohens_d_paired": d_paired,
        "rank_biserial": rbc,
    })

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError("No valid paired features found after filtering.")

# =========================================================
# FDR CORRECTION
# =========================================================
# FDR for t-test
mask_t = results_df["p_ttest"].notna()
q_t = np.full(len(results_df), np.nan)
if mask_t.sum() > 0:
    _, qvals_t, _, _ = multipletests(results_df.loc[mask_t, "p_ttest"], alpha=alpha, method="fdr_bh")
    q_t[mask_t] = qvals_t
results_df["q_ttest_fdr"] = q_t

# FDR for Wilcoxon
mask_w = results_df["p_wilcoxon"].notna()
q_w = np.full(len(results_df), np.nan)
if mask_w.sum() > 0:
    _, qvals_w, _, _ = multipletests(results_df.loc[mask_w, "p_wilcoxon"], alpha=alpha, method="fdr_bh")
    q_w[mask_w] = qvals_w
results_df["q_wilcoxon_fdr"] = q_w

# Rank results
results_df = results_df.sort_values(
    by=["q_wilcoxon_fdr", "q_ttest_fdr", "abs_mean_delta"],
    ascending=[True, True, False]
) if "abs_mean_delta" in results_df.columns else results_df

results_df["abs_mean_delta"] = results_df["mean_delta"].abs()
results_df = results_df.sort_values(
    by=["q_wilcoxon_fdr", "q_ttest_fdr", "abs_mean_delta"],
    ascending=[True, True, False]
).reset_index(drop=True)

# =========================================================
# SAVE FULL + SIGNIFICANT
# =========================================================
full_out = os.path.join(output_dir, "paired_pre_post_radiomics_all.csv")
results_df.to_csv(full_out, index=False)

sig_df = results_df[
    (results_df["q_ttest_fdr"] < alpha) | (results_df["q_wilcoxon_fdr"] < alpha)
].copy()

sig_out = os.path.join(output_dir, "paired_pre_post_radiomics_significant_fdr.csv")
sig_df.to_csv(sig_out, index=False)

print("Saved full results to:", full_out)
print("Saved significant results to:", sig_out)
print("N significant features:", len(sig_df))

# =========================================================
# OPTIONAL: DELTA FEATURES TABLE
# =========================================================
delta_df = X_df[[c for c in X_df.columns if c in ["cohort", "id", "session_pre", "session_post", "subject_code"]]].copy()

for base in feature_bases:
    pre_col = f"{base}_pre"
    post_col = f"{base}_post"
    delta_df[f"{base}_delta"] = X_df[post_col] - X_df[pre_col]

delta_out = os.path.join(output_dir, "radiomics_delta_features.csv")
delta_df.to_csv(delta_out, index=False)
print("Saved delta feature table to:", delta_out)

# =========================================================
# OPTIONAL: CORRELATION WITH OUTCOME
# =========================================================
if merged_y is not None:
    corr_df = delta_df.merge(merged_y, on=["cohort", "id"], how="inner")

    corr_results = []
    delta_cols = [c for c in corr_df.columns if c.endswith("_delta")]

    for col in delta_cols:
        sub = corr_df[[col, outcome_col]].copy().dropna()
        if len(sub) < 3:
            continue

        x = sub[col].to_numpy(dtype=float)
        yy = sub[outcome_col].to_numpy(dtype=float)

        roi, short_feature = parse_roi_and_feature(col[:-6])  # remove "_delta"

        try:
            r_p, p_p = pearsonr(x, yy)
        except Exception:
            r_p, p_p = np.nan, np.nan

        try:
            r_s, p_s = spearmanr(x, yy)
        except Exception:
            r_s, p_s = np.nan, np.nan

        corr_results.append({
            "roi": roi,
            "delta_feature": col,
            "feature_name": short_feature,
            "n": len(sub),
            "pearson_r": r_p,
            "pearson_p": p_p,
            "spearman_rho": r_s,
            "spearman_p": p_s,
        })

    corr_results_df = pd.DataFrame(corr_results)

    if not corr_results_df.empty:
        # FDR for Spearman
        mask_s = corr_results_df["spearman_p"].notna()
        q_s = np.full(len(corr_results_df), np.nan)
        if mask_s.sum() > 0:
            _, qvals_s, _, _ = multipletests(corr_results_df.loc[mask_s, "spearman_p"], alpha=alpha, method="fdr_bh")
            q_s[mask_s] = qvals_s
        corr_results_df["spearman_q_fdr"] = q_s

        # FDR for Pearson
        mask_p = corr_results_df["pearson_p"].notna()
        q_p = np.full(len(corr_results_df), np.nan)
        if mask_p.sum() > 0:
            _, qvals_p, _, _ = multipletests(corr_results_df.loc[mask_p, "pearson_p"], alpha=alpha, method="fdr_bh")
            q_p[mask_p] = qvals_p
        corr_results_df["pearson_q_fdr"] = q_p

        corr_results_df = corr_results_df.sort_values(
            by=["spearman_q_fdr", "pearson_q_fdr"],
            ascending=[True, True]
        ).reset_index(drop=True)

        corr_all_out = os.path.join(output_dir, "delta_radiomics_vs_outcome_all.csv")
        corr_results_df.to_csv(corr_all_out, index=False)

        corr_sig_df = corr_results_df[
            (corr_results_df["spearman_q_fdr"] < alpha) | (corr_results_df["pearson_q_fdr"] < alpha)
        ].copy()

        corr_sig_out = os.path.join(output_dir, "delta_radiomics_vs_outcome_significant_fdr.csv")
        corr_sig_df.to_csv(corr_sig_out, index=False)

        print("Saved outcome-correlation results to:", corr_all_out)
        print("Saved significant outcome correlations to:", corr_sig_out)
        print("N significant outcome correlations:", len(corr_sig_df))
    else:
        print("No valid delta-vs-outcome correlations could be computed.")