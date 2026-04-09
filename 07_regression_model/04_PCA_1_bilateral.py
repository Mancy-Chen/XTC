import os
import re
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, wilcoxon, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# =========================================================
# SETTINGS
# =========================================================
input_dir = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC"
output_root = "/data/projects/CSC/code/XTC/07_regression_model/Output/roi_pca_structural_change_bilateral"
os.makedirs(output_root, exist_ok=True)

x_path = f"{input_dir}/X_pre_delta_with_id_filteredQC.csv"
y_path = f"{input_dir}/y_with_id_filteredQC.csv"

outcome_col = "vwrec"
alpha = 0.05

# bilateral anatomical groups
target_groups = {
    "Hippocampus": ["Left_Hippocampus", "Right_Hippocampus"],
    "Thalamus": ["Left_Thalamus", "Right_Thalamus"],
}

all_target_rois = [roi for rois in target_groups.values() for roi in rois]

# obvious volume/size-related patterns to exclude in "non_volume"
volume_like_patterns = [
    "MeshVolume",
    "VoxelVolume",
    "SurfaceArea",
    "Maximum3DDiameter",
    "Maximum2DDiameter",
    "MajorAxisLength",
    "MinorAxisLength",
    "LeastAxisLength",
    "Elongation",
    "Flatness",
    "Sphericity",
    "Compactness",
    "Volume",
]

analysis_configs = {
    "shape_only": {
        "include_keywords": ["shape"],
        "exclude_volume_like": False,
    },
    "firstorder_only": {
        "include_keywords": ["firstorder"],
        "exclude_volume_like": False,
    },
    "non_volume": {
        "include_keywords": ["shape", "firstorder"],
        "exclude_volume_like": True,
    },
}

# =========================================================
# LOAD DATA
# =========================================================
X_df = pd.read_csv(x_path)
y_df = pd.read_csv(y_path)

print("Loaded X:", X_df.shape)
print("Loaded y:", y_df.shape)

if outcome_col not in y_df.columns:
    raise ValueError(f"{outcome_col} not found in y file.")

y_merge = y_df[["cohort", "id", outcome_col]].copy()
y_merge[outcome_col] = pd.to_numeric(y_merge[outcome_col], errors="coerce")

df = X_df.merge(y_merge, on=["cohort", "id"], how="left")
print("Merged df:", df.shape)

# =========================================================
# BASIC CLEANING
# =========================================================
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.replace(",", ".", regex=False).str.strip()

id_like_cols = {"subject_code"}
for c in df.columns:
    if c not in id_like_cols:
        df[c] = pd.to_numeric(df[c], errors="ignore")

# =========================================================
# RECONSTRUCT POST FROM PRE + DELTA
# =========================================================
all_cols = df.columns.tolist()
pre_cols = [c for c in all_cols if c.endswith("_pre")]
delta_cols = [c for c in all_cols if c.endswith("_delta")]

pre_bases = {c[:-4] for c in pre_cols}
delta_bases = {c[:-6] for c in delta_cols}
common_bases = sorted(pre_bases & delta_bases)

for base in common_bases:
    pre_col = f"{base}_pre"
    delta_col = f"{base}_delta"
    post_col = f"{base}_post"

    df[pre_col] = pd.to_numeric(df[pre_col], errors="coerce")
    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
    df[post_col] = df[pre_col] + df[delta_col]

print("Number of common pre/delta bases:", len(common_bases))

# =========================================================
# HELPERS
# =========================================================
def parse_roi_and_feature(base_name: str):
    for roi in all_target_rois:
        prefix = roi + "_"
        if base_name.startswith(prefix):
            return roi, base_name[len(prefix):]
    return None, base_name

def feature_matches_keywords(base_name: str, keywords):
    return any(k.lower() in base_name.lower() for k in keywords)

def is_volume_like(base_name: str):
    return any(p.lower() in base_name.lower() for p in volume_like_patterns)

def get_feature_bases_for_analysis(base_list, include_keywords, exclude_volume_like=False):
    selected = []
    for b in base_list:
        roi, short_feature = parse_roi_and_feature(b)
        if roi is None:
            continue
        if not feature_matches_keywords(b, include_keywords):
            continue
        if exclude_volume_like and is_volume_like(b):
            continue
        selected.append(b)
    return selected

def get_group_feature_bases(group_rois, base_list):
    return [b for b in base_list if any(b.startswith(roi + "_") for roi in group_rois)]

def clean_complete_pairs(df_sub, cols_pre, cols_post):
    tmp = df_sub[cols_pre + cols_post].copy()
    complete = ~tmp.isna().any(axis=1)
    return complete

def paired_effect_size(pre, post):
    diff = post - pre
    sd = np.std(diff, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return np.mean(diff) / sd

def rank_biserial(pre, post):
    try:
        stat, _ = wilcoxon(post, pre, zero_method="wilcox", alternative="two-sided")
        diff = post - pre
        diff = diff[~np.isnan(diff)]
        diff = diff[diff != 0]
        n = len(diff)
        if n == 0:
            return np.nan
        total_rank_sum = n * (n + 1) / 2
        return 1 - (2 * stat / total_rank_sum)
    except Exception:
        return np.nan

def safe_pearson(x, y):
    try:
        return pearsonr(x, y)
    except Exception:
        return (np.nan, np.nan)

def safe_spearman(x, y):
    try:
        return spearmanr(x, y)
    except Exception:
        return (np.nan, np.nan)

def pca_scores_from_prepost(df_in, feature_bases):
    cols_pre = [f"{b}_pre" for b in feature_bases]
    cols_post = [f"{b}_post" for b in feature_bases]

    complete = clean_complete_pairs(df_in, cols_pre, cols_post)
    sub = df_in.loc[complete, ["cohort", "id", outcome_col] + cols_pre + cols_post].copy()

    if sub.shape[0] < 5:
        return None, None, np.nan

    X_pre = sub[cols_pre].to_numpy(dtype=float)
    X_post = sub[cols_post].to_numpy(dtype=float)

    X_both = np.vstack([X_pre, X_post])

    scaler = StandardScaler()
    X_both_scaled = scaler.fit_transform(X_both)

    pca = PCA(n_components=1, random_state=42)
    PC1_both = pca.fit_transform(X_both_scaled).ravel()

    n = X_pre.shape[0]
    PC1_pre = PC1_both[:n]
    PC1_post = PC1_both[n:]
    PC1_delta = PC1_post - PC1_pre

    out_df = sub[["cohort", "id", outcome_col]].copy()
    out_df["PC1_pre"] = PC1_pre
    out_df["PC1_post"] = PC1_post
    out_df["PC1_delta"] = PC1_delta

    loadings = pca.components_[0]
    feature_names_short = [re.sub(r"_pre$", "", c) for c in cols_pre]
    loadings_df = pd.DataFrame({
        "feature": feature_names_short,
        "loading_PC1": loadings,
        "abs_loading_PC1": np.abs(loadings),
    }).sort_values("abs_loading_PC1", ascending=False).reset_index(drop=True)

    explained_var = pca.explained_variance_ratio_[0]
    return out_df, loadings_df, explained_var

def find_volume_columns_for_group(df_in, group_rois):
    rows = []
    for roi in group_rois:
        pre_col = None
        delta_col = None
        candidates = []

        for col in df_in.columns:
            if col.startswith(roi + "_") and (col.endswith("_pre") or col.endswith("_delta")):
                if any(v.lower() in col.lower() for v in ["meshvolume", "voxelvolume", "volume"]):
                    candidates.append(col)

        def score(col):
            c = col.lower()
            if "meshvolume" in c:
                return 0
            if "voxelvolume" in c:
                return 1
            if "volume" in c:
                return 2
            return 99

        candidates = sorted(candidates, key=score)

        for c in candidates:
            if c.endswith("_pre") and pre_col is None:
                pre_col = c
            if c.endswith("_delta") and delta_col is None:
                delta_col = c

        rows.append({
            "roi": roi,
            "volume_pre_col": pre_col,
            "volume_delta_col": delta_col,
        })

    return pd.DataFrame(rows)

# =========================================================
# RUN ANALYSES
# =========================================================
for analysis_name, cfg in analysis_configs.items():
    print("\n" + "=" * 70)
    print(f"Running analysis: {analysis_name}")
    print("=" * 70)

    output_dir = os.path.join(output_root, analysis_name)
    os.makedirs(output_dir, exist_ok=True)

    feature_bases = get_feature_bases_for_analysis(
        common_bases,
        include_keywords=cfg["include_keywords"],
        exclude_volume_like=cfg["exclude_volume_like"],
    )

    print("Number of selected feature bases:", len(feature_bases))

    all_group_pc_dfs = []
    paired_results = []
    corr_results = []
    volume_corr_results = []
    explained_variance_rows = []

    for group_name, group_rois in target_groups.items():
        group_bases = get_group_feature_bases(group_rois, feature_bases)
        print(f"{group_name}: {len(group_bases)} feature bases across {group_rois}")

        if len(group_bases) < 2:
            print(f"Skipping {group_name}: not enough features.")
            continue

        group_pc_df, loadings_df, explained_var = pca_scores_from_prepost(df, group_bases)

        if group_pc_df is None:
            print(f"Skipping {group_name}: not enough complete cases.")
            continue

        group_pc_df["structure"] = group_name
        all_group_pc_dfs.append(group_pc_df)

        loadings_df.to_csv(
            os.path.join(output_dir, f"{group_name}_PC1_loadings.csv"),
            index=False
        )

        explained_variance_rows.append({
            "analysis": analysis_name,
            "structure": group_name,
            "rois_included": ";".join(group_rois),
            "n_subjects": group_pc_df.shape[0],
            "n_features_in_pca": len(group_bases),
            "PC1_explained_variance_ratio": explained_var,
        })

        # paired test
        pre = group_pc_df["PC1_pre"].to_numpy(dtype=float)
        post = group_pc_df["PC1_post"].to_numpy(dtype=float)
        delta = group_pc_df["PC1_delta"].to_numpy(dtype=float)

        t_stat, p_t = ttest_rel(post, pre, nan_policy="omit")
        try:
            w_stat, p_w = wilcoxon(post, pre, zero_method="wilcox", alternative="two-sided")
        except Exception:
            w_stat, p_w = np.nan, np.nan

        paired_results.append({
            "analysis": analysis_name,
            "structure": group_name,
            "rois_included": ";".join(group_rois),
            "n": len(group_pc_df),
            "PC1_explained_variance_ratio": explained_var,
            "mean_PC1_pre": np.mean(pre),
            "mean_PC1_post": np.mean(post),
            "mean_PC1_delta": np.mean(delta),
            "median_PC1_delta": np.median(delta),
            "t_stat": t_stat,
            "p_ttest": p_t,
            "w_stat": w_stat,
            "p_wilcoxon": p_w,
            "cohens_d_paired": paired_effect_size(pre, post),
            "rank_biserial": rank_biserial(pre, post),
        })

        # delta PC1 vs cognition
        sub_corr = group_pc_df[["PC1_delta", outcome_col]].dropna().copy()
        if sub_corr.shape[0] >= 3:
            x = sub_corr["PC1_delta"].to_numpy(dtype=float)
            y = sub_corr[outcome_col].to_numpy(dtype=float)

            pear_r, pear_p = safe_pearson(x, y)
            spear_r, spear_p = safe_spearman(x, y)

            corr_results.append({
                "analysis": analysis_name,
                "structure": group_name,
                "rois_included": ";".join(group_rois),
                "n": len(sub_corr),
                "pearson_r": pear_r,
                "pearson_p": pear_p,
                "spearman_rho": spear_r,
                "spearman_p": spear_p,
            })

        # delta PC1 vs delta volume, per side
        volume_cols_df = find_volume_columns_for_group(df, group_rois)

        for _, row in volume_cols_df.iterrows():
            roi = row["roi"]
            vol_pre_col = row["volume_pre_col"]
            vol_delta_col = row["volume_delta_col"]

            if pd.isna(vol_delta_col) or vol_delta_col is None:
                continue

            tmp = group_pc_df[["cohort", "id", "PC1_delta"]].merge(
                df[["cohort", "id", vol_delta_col]],
                on=["cohort", "id"],
                how="left"
            ).dropna()

            if tmp.shape[0] >= 3:
                x = tmp["PC1_delta"].to_numpy(dtype=float)
                y = pd.to_numeric(tmp[vol_delta_col], errors="coerce").to_numpy(dtype=float)

                pear_r, pear_p = safe_pearson(x, y)
                spear_r, spear_p = safe_spearman(x, y)

                volume_corr_results.append({
                    "analysis": analysis_name,
                    "structure": group_name,
                    "roi_volume_side": roi,
                    "volume_pre_col": vol_pre_col,
                    "volume_delta_col": vol_delta_col,
                    "n": len(tmp),
                    "pearson_r": pear_r,
                    "pearson_p": pear_p,
                    "spearman_rho": spear_r,
                    "spearman_p": spear_p,
                })

    if len(all_group_pc_dfs) == 0:
        print(f"No bilateral PCA results for {analysis_name}.")
        continue

    # save PC scores
    all_group_pc_df = pd.concat(all_group_pc_dfs, axis=0, ignore_index=True)
    all_group_pc_df.to_csv(os.path.join(output_dir, "structure_pc_scores_long.csv"), index=False)

    group_pc_wide = all_group_pc_df.pivot_table(
        index=["cohort", "id", outcome_col],
        columns="structure",
        values=["PC1_pre", "PC1_post", "PC1_delta"]
    )
    group_pc_wide.columns = [f"{metric}_{structure}" for metric, structure in group_pc_wide.columns]
    group_pc_wide = group_pc_wide.reset_index()
    group_pc_wide.to_csv(os.path.join(output_dir, "structure_pc_scores_wide.csv"), index=False)

    # paired tests
    paired_df = pd.DataFrame(paired_results)
    for pcol, qcol in [("p_ttest", "q_ttest_fdr"), ("p_wilcoxon", "q_wilcoxon_fdr")]:
        mask = paired_df[pcol].notna()
        qvals = np.full(len(paired_df), np.nan)
        if mask.sum() > 0:
            _, q, _, _ = multipletests(paired_df.loc[mask, pcol], alpha=alpha, method="fdr_bh")
            qvals[mask] = q
        paired_df[qcol] = qvals

    paired_df = paired_df.sort_values(
        by=["q_wilcoxon_fdr", "q_ttest_fdr"],
        ascending=[True, True]
    ).reset_index(drop=True)

    paired_df.to_csv(os.path.join(output_dir, "structure_pc1_paired_tests.csv"), index=False)

    # explained variance
    ev_df = pd.DataFrame(explained_variance_rows)
    ev_df.to_csv(os.path.join(output_dir, "structure_pc1_explained_variance.csv"), index=False)

    # cognition correlations
    corr_df = pd.DataFrame(corr_results)
    if not corr_df.empty:
        for pcol, qcol in [("pearson_p", "pearson_q_fdr"), ("spearman_p", "spearman_q_fdr")]:
            mask = corr_df[pcol].notna()
            qvals = np.full(len(corr_df), np.nan)
            if mask.sum() > 0:
                _, q, _, _ = multipletests(corr_df.loc[mask, pcol], alpha=alpha, method="fdr_bh")
                qvals[mask] = q
            corr_df[qcol] = qvals

        corr_df = corr_df.sort_values(
            by=["spearman_q_fdr", "pearson_q_fdr"],
            ascending=[True, True]
        ).reset_index(drop=True)

    corr_df.to_csv(os.path.join(output_dir, "structure_pc1_vs_cognition.csv"), index=False)

    # volume-change correlations
    volcorr_df = pd.DataFrame(volume_corr_results)
    if not volcorr_df.empty:
        for pcol, qcol in [("pearson_p", "pearson_q_fdr"), ("spearman_p", "spearman_q_fdr")]:
            mask = volcorr_df[pcol].notna()
            qvals = np.full(len(volcorr_df), np.nan)
            if mask.sum() > 0:
                _, q, _, _ = multipletests(volcorr_df.loc[mask, pcol], alpha=alpha, method="fdr_bh")
                qvals[mask] = q
            volcorr_df[qcol] = qvals

        volcorr_df = volcorr_df.sort_values(
            by=["spearman_q_fdr", "pearson_q_fdr"],
            ascending=[True, True]
        ).reset_index(drop=True)

    volcorr_df.to_csv(os.path.join(output_dir, "structure_pc1_vs_volume_change.csv"), index=False)

    print(f"\nFinished {analysis_name}")
    print("Saved to:", output_dir)
    print("\nPaired tests:")
    print(paired_df)
    print("\nCognition correlations:")
    print(corr_df if not corr_df.empty else "None")
    print("\nVolume-change correlations:")
    print(volcorr_df if not volcorr_df.empty else "None")

print("\nAll bilateral analyses completed.")
print("Output root:", output_root)