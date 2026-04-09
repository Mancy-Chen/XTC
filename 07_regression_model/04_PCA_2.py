# Mingshi Chen 25-03-2026
# Mixed-effects model on ROI shape-PC1
# Model: PC1 ~ session + sex + BrainSegVol + (1 | subject_id)

import os
import re
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

# =========================================================
# SETTINGS
# =========================================================
input_dir = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC"
output_dir = "/data/projects/CSC/code/XTC/07_regression_model/Output/shape_pc1_mixed_model"
os.makedirs(output_dir, exist_ok=True)

# Radiomics paired file
x_path = f"{input_dir}/X_pre_delta_with_id_filteredQC.csv"

# Covariate / volume file
cov_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/hippocampus_thalamus_with_globalvolumes_with_sex_age_QC_excluded_large_hippocampus.xlsx"
cov_sheet = "QC_excluded_data"

# Session labels used in the covariate sheet
pre_session_label = "sessie1"
post_session_label = "sessie3"

# ROIs
target_rois = [
    "Left_Hippocampus",
    "Right_Hippocampus",
    "Left_Thalamus",
    "Right_Thalamus",
]

# Shape-only PCA
feature_keywords = ["shape"]

# Optional: exclude obvious size-driven shape features
exclude_volume_like = False

volume_like_patterns = [
    "MeshVolume",
    "VoxelVolume",
    "SurfaceArea",
    "Maximum3DDiameter",
    "Maximum2DDiameter",
    "MajorAxisLength",
    "MinorAxisLength",
    "LeastAxisLength",
    "Volume",
]

# =========================================================
# LOAD DATA
# =========================================================
X_df = pd.read_csv(x_path)
cov_df = pd.read_excel(cov_path, sheet_name=cov_sheet)

print("Loaded radiomics:", X_df.shape)
print("Loaded covariates:", cov_df.shape)
print("Covariate columns:", cov_df.columns.tolist())

# =========================================================
# CLEAN DATA
# =========================================================
for df in [X_df, cov_df]:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False).str.strip()

# Convert likely numeric columns
for c in X_df.columns:
    if c != "subject_code":
        X_df[c] = pd.to_numeric(X_df[c], errors="ignore")

for c in cov_df.columns:
    if c not in ["folder_name", "cohort", "subject_id", "session"]:
        cov_df[c] = pd.to_numeric(cov_df[c], errors="coerce")

# Keep only sessie1 and sessie3
cov_df = cov_df[cov_df["session"].isin([pre_session_label, post_session_label])].copy()

# =========================================================
# RECONSTRUCT POST FROM PRE + DELTA
# =========================================================
all_cols = X_df.columns.tolist()
pre_cols = [c for c in all_cols if c.endswith("_pre")]
delta_cols = [c for c in all_cols if c.endswith("_delta")]

pre_bases = {c[:-4] for c in pre_cols}
delta_bases = {c[:-6] for c in delta_cols}
common_bases = sorted(pre_bases & delta_bases)

for base in common_bases:
    pre_col = f"{base}_pre"
    delta_col = f"{base}_delta"
    post_col = f"{base}_post"

    X_df[pre_col] = pd.to_numeric(X_df[pre_col], errors="coerce")
    X_df[delta_col] = pd.to_numeric(X_df[delta_col], errors="coerce")
    X_df[post_col] = X_df[pre_col] + X_df[delta_col]

print("Number of common pre/delta bases:", len(common_bases))

# =========================================================
# HELPERS
# =========================================================
def parse_roi_and_feature(base_name: str):
    for roi in target_rois:
        prefix = roi + "_"
        if base_name.startswith(prefix):
            return roi, base_name[len(prefix):]
    return None, base_name

def is_volume_like(base_name: str):
    return any(p.lower() in base_name.lower() for p in volume_like_patterns)

def get_shape_feature_bases(base_list):
    selected = []
    for b in base_list:
        roi, short_feature = parse_roi_and_feature(b)
        if roi is None:
            continue
        if not any(k in b for k in feature_keywords):
            continue
        if exclude_volume_like and is_volume_like(b):
            continue
        selected.append(b)
    return selected

def get_roi_feature_bases(roi_name, base_list):
    return [b for b in base_list if b.startswith(roi_name + "_")]

def compute_roi_pc1_long(df_in, roi_name, roi_feature_bases):
    cols_pre = [f"{b}_pre" for b in roi_feature_bases]
    cols_post = [f"{b}_post" for b in roi_feature_bases]

    tmp = df_in[["cohort", "id"] + cols_pre + cols_post].copy()
    complete = ~tmp[cols_pre + cols_post].isna().any(axis=1)
    tmp = tmp.loc[complete].copy()

    if tmp.shape[0] < 5:
        return None, None, np.nan

    X_pre = tmp[cols_pre].to_numpy(dtype=float)
    X_post = tmp[cols_post].to_numpy(dtype=float)

    X_both = np.vstack([X_pre, X_post])

    scaler = StandardScaler()
    X_both_scaled = scaler.fit_transform(X_both)

    pca = PCA(n_components=1, random_state=42)
    PC1_both = pca.fit_transform(X_both_scaled).ravel()

    n = X_pre.shape[0]
    PC1_pre = PC1_both[:n]
    PC1_post = PC1_both[n:]

    long_pre = tmp[["cohort", "id"]].copy()
    long_pre["session"] = pre_session_label
    long_pre["PC1"] = PC1_pre
    long_pre["roi"] = roi_name

    long_post = tmp[["cohort", "id"]].copy()
    long_post["session"] = post_session_label
    long_post["PC1"] = PC1_post
    long_post["roi"] = roi_name

    long_df = pd.concat([long_pre, long_post], axis=0, ignore_index=True)

    roman_map = {"1": "I", "2": "II", "3": "III", "4": "IV", 1: "I", 2: "II", 3: "III", 4: "IV"}

    long_df["cohort"] = long_df["cohort"].astype(str).str.strip()
    long_df["subject_id"] = (
        long_df["cohort"].map(roman_map) +
        long_df["id"].astype(int).astype(str).str.zfill(3)
    )

    loadings_df = pd.DataFrame({
        "feature": [re.sub(r"_pre$", "", c) for c in cols_pre],
        "loading_PC1": pca.components_[0],
        "abs_loading_PC1": np.abs(pca.components_[0]),
    }).sort_values("abs_loading_PC1", ascending=False).reset_index(drop=True)

    explained_var = pca.explained_variance_ratio_[0]
    return long_df, loadings_df, explained_var

# =========================================================
# BUILD SHAPE-ONLY ROI PC1
# =========================================================
shape_feature_bases = get_shape_feature_bases(common_bases)
print("Number of shape feature bases:", len(shape_feature_bases))

all_long = []
explained_rows = []

for roi in target_rois:
    roi_bases = get_roi_feature_bases(roi, shape_feature_bases)
    print(f"{roi}: {len(roi_bases)} shape features")

    if len(roi_bases) < 2:
        print(f"Skipping {roi}: not enough features.")
        continue

    roi_long, loadings_df, explained_var = compute_roi_pc1_long(X_df, roi, roi_bases)

    if roi_long is None:
        print(f"Skipping {roi}: not enough complete paired subjects.")
        continue

    all_long.append(roi_long)

    loadings_df.to_csv(
        os.path.join(output_dir, f"{roi}_shape_PC1_loadings.csv"),
        index=False
    )

    explained_rows.append({
        "roi": roi,
        "n_features": len(roi_bases),
        "n_subjects": roi_long["subject_id"].nunique(),
        "PC1_explained_variance_ratio": explained_var,
    })

if len(all_long) == 0:
    raise ValueError("No ROI PC1 data generated.")

pc1_long_df = pd.concat(all_long, axis=0, ignore_index=True)
pc1_long_df.to_csv(os.path.join(output_dir, "shape_pc1_long.csv"), index=False)

ev_df = pd.DataFrame(explained_rows)
ev_df.to_csv(os.path.join(output_dir, "shape_pc1_explained_variance.csv"), index=False)

print("Saved shape PC1 long data and explained variance.")

# =========================================================
# PREPARE COVARIATES
# =========================================================
required_cov_cols = [
    "cohort", "subject_id", "session", "sex", "BrainSegVol",
    "Left_Hippocampus", "Right_Hippocampus", "Left_Thalamus", "Right_Thalamus"
]
missing_cov_cols = [c for c in required_cov_cols if c not in cov_df.columns]
if missing_cov_cols:
    raise ValueError(f"Missing required columns in covariate file: {missing_cov_cols}")

cov_keep = [
    "cohort", "subject_id", "session", "sex", "age", "BrainSegVol",
    "Left_Hippocampus", "Right_Hippocampus", "Left_Thalamus", "Right_Thalamus"
]
cov_keep = [c for c in cov_keep if c in cov_df.columns]

cov_sub = cov_df[cov_keep].copy()

# --- STANDARDIZE MERGE KEYS ---
# make everything string-based to avoid int/object mismatch
pc1_long_df["cohort"] = pc1_long_df["cohort"].astype(str).str.strip()
pc1_long_df["subject_id"] = pc1_long_df["subject_id"].astype(str).str.strip()
pc1_long_df["session"] = pc1_long_df["session"].astype(str).str.strip()

cov_sub["cohort"] = cov_sub["cohort"].astype(str).str.strip()
cov_sub["subject_id"] = cov_sub["subject_id"].astype(str).str.strip()
cov_sub["session"] = cov_sub["session"].astype(str).str.strip()

# keep only sessions of interest
cov_sub = cov_sub[cov_sub["session"].isin([pre_session_label, post_session_label])].copy()

# numeric covariates
cov_sub["sex"] = pd.to_numeric(cov_sub["sex"], errors="coerce")
cov_sub["BrainSegVol"] = pd.to_numeric(cov_sub["BrainSegVol"], errors="coerce")
if "age" in cov_sub.columns:
    cov_sub["age"] = pd.to_numeric(cov_sub["age"], errors="coerce")

# session categorical with sessie1 as reference
cov_sub["session"] = pd.Categorical(
    cov_sub["session"],
    categories=[pre_session_label, post_session_label],
    ordered=True
)

# =========================================================
# MERGE PC1 WITH COVARIATES
# =========================================================
# Keep only fields needed
cov_keep = [
    "subject_id", "session", "sex", "age", "BrainSegVol",
    "Left_Hippocampus", "Right_Hippocampus", "Left_Thalamus", "Right_Thalamus"
]
cov_keep = [c for c in cov_keep if c in cov_df.columns]

cov_sub = cov_df[cov_keep].copy()

pc1_long_df["subject_id"] = pc1_long_df["subject_id"].astype(str).str.strip()
pc1_long_df["session"] = pc1_long_df["session"].astype(str).str.strip()

cov_sub["subject_id"] = cov_sub["subject_id"].astype(str).str.strip()
cov_sub["session"] = cov_sub["session"].astype(str).str.strip()

cov_sub = cov_sub[cov_sub["session"].isin([pre_session_label, post_session_label])].copy()

cov_sub["sex"] = pd.to_numeric(cov_sub["sex"], errors="coerce")
cov_sub["BrainSegVol"] = pd.to_numeric(cov_sub["BrainSegVol"], errors="coerce")
if "age" in cov_sub.columns:
    cov_sub["age"] = pd.to_numeric(cov_sub["age"], errors="coerce")

cov_sub["session"] = pd.Categorical(
    cov_sub["session"],
    categories=[pre_session_label, post_session_label],
    ordered=True
)

model_df = pc1_long_df.merge(
    cov_sub,
    on=["subject_id", "session"],
    how="left"
)

print("Merged model df:", model_df.shape)
print(model_df.head())
print(model_df[["subject_id", "session", "sex", "BrainSegVol"]].head(20))
# =========================================================
# FIT MIXED MODELS ROI BY ROI
# =========================================================
results = []

for roi in target_rois:
    sub = model_df[model_df["roi"] == roi].copy()

    # drop missing model vars
    sub = sub.dropna(subset=["PC1", "session", "sex", "BrainSegVol", "subject_id"]).copy()

    if sub["subject_id"].nunique() < 10:
        print(f"Skipping {roi}: too few subjects.")
        continue

    # same framework as your volume model
    formula = "PC1 ~ C(session) + sex + BrainSegVol"

    try:
        model = smf.mixedlm(
            formula=formula,
            data=sub,
            groups=sub["subject_id"]
        )
        fit = model.fit(reml=False, method="lbfgs", maxiter=200)

        # Save full summary
        with open(os.path.join(output_dir, f"{roi}_mixedlm_summary.txt"), "w") as f:
            f.write(str(fit.summary()))

        # Extract session effect: sessie3 - sessie1
        term = f"C(session)[T.{post_session_label}]"

        est = fit.params.get(term, np.nan)
        pval = fit.pvalues.get(term, np.nan)
        tval = fit.tvalues.get(term, np.nan)

        conf_int = fit.conf_int()
        if term in conf_int.index:
            ci_low = conf_int.loc[term, 0]
            ci_high = conf_int.loc[term, 1]
        else:
            ci_low, ci_high = np.nan, np.nan

        results.append({
            "roi": roi,
            "n_rows": len(sub),
            "n_subjects": sub["subject_id"].nunique(),
            "estimate_session3_minus_session1": est,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "t_value": tval,
            "p_value": pval,
        })

        print(f"\n================ {roi} ================\n")
        print(fit.summary())

    except Exception as e:
        print(f"Model failed for {roi}: {e}")
        results.append({
            "roi": roi,
            "n_rows": len(sub),
            "n_subjects": sub["subject_id"].nunique(),
            "estimate_session3_minus_session1": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "error": str(e),
        })

# =========================================================
# SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "shape_pc1_mixedlm_results.csv"), index=False)

print("\nSaved mixed model results to:")
print(os.path.join(output_dir, "shape_pc1_mixedlm_results.csv"))
print(results_df)