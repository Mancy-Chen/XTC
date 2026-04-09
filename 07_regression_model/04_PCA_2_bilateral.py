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
output_dir = "/data/projects/CSC/code/XTC/07_regression_model/Output/shape_pc1_mixed_model_bilateral"
os.makedirs(output_dir, exist_ok=True)

x_path = f"{input_dir}/X_pre_delta_with_id_filteredQC.csv"

cov_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/hippocampus_thalamus_with_globalvolumes_with_sex_age_QC_excluded_large_hippocampus.xlsx"
cov_sheet = "QC_excluded_data"

pre_session_label = "sessie1"
post_session_label = "sessie3"

# bilateral groups
target_groups = {
    "Hippocampus": ["Left_Hippocampus", "Right_Hippocampus"],
    "Thalamus": ["Left_Thalamus", "Right_Thalamus"],
}
all_target_rois = [r for rois in target_groups.values() for r in rois]

feature_keywords = ["shape"]
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
for df_ in [X_df, cov_df]:
    for c in df_.columns:
        if df_[c].dtype == object:
            df_[c] = df_[c].astype(str).str.replace(",", ".", regex=False).str.strip()

for c in X_df.columns:
    if c != "subject_code":
        X_df[c] = pd.to_numeric(X_df[c], errors="ignore")

for c in cov_df.columns:
    if c not in ["folder_name", "cohort", "subject_id", "session"]:
        cov_df[c] = pd.to_numeric(cov_df[c], errors="coerce")

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
    for roi in all_target_rois:
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

def get_group_feature_bases(group_rois, base_list):
    return [b for b in base_list if any(b.startswith(roi + "_") for roi in group_rois)]

def compute_group_pc1_long(df_in, group_name, group_rois, group_feature_bases):
    cols_pre = [f"{b}_pre" for b in group_feature_bases]
    cols_post = [f"{b}_post" for b in group_feature_bases]

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
    long_pre["structure"] = group_name

    long_post = tmp[["cohort", "id"]].copy()
    long_post["session"] = post_session_label
    long_post["PC1"] = PC1_post
    long_post["structure"] = group_name

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
# BUILD SHAPE-ONLY BILATERAL PC1
# =========================================================
shape_feature_bases = get_shape_feature_bases(common_bases)
print("Number of shape feature bases:", len(shape_feature_bases))

all_long = []
explained_rows = []

for group_name, group_rois in target_groups.items():
    group_bases = get_group_feature_bases(group_rois, shape_feature_bases)
    print(f"{group_name}: {len(group_bases)} shape features across {group_rois}")

    if len(group_bases) < 2:
        print(f"Skipping {group_name}: not enough features.")
        continue

    group_long, loadings_df, explained_var = compute_group_pc1_long(
        X_df, group_name, group_rois, group_bases
    )

    if group_long is None:
        print(f"Skipping {group_name}: not enough complete paired subjects.")
        continue

    all_long.append(group_long)

    loadings_df.to_csv(
        os.path.join(output_dir, f"{group_name}_shape_PC1_loadings.csv"),
        index=False
    )

    explained_rows.append({
        "structure": group_name,
        "rois_included": ";".join(group_rois),
        "n_features": len(group_bases),
        "n_subjects": group_long["subject_id"].nunique(),
        "PC1_explained_variance_ratio": explained_var,
    })

if len(all_long) == 0:
    raise ValueError("No bilateral PC1 data generated.")

pc1_long_df = pd.concat(all_long, axis=0, ignore_index=True)
pc1_long_df.to_csv(os.path.join(output_dir, "shape_pc1_long_bilateral.csv"), index=False)

ev_df = pd.DataFrame(explained_rows)
ev_df.to_csv(os.path.join(output_dir, "shape_pc1_explained_variance_bilateral.csv"), index=False)

print("Saved bilateral shape PC1 long data and explained variance.")

# =========================================================
# PREPARE COVARIATES
# =========================================================
required_cov_cols = [
    "subject_id", "session", "sex", "BrainSegVol",
    "Left_Hippocampus", "Right_Hippocampus", "Left_Thalamus", "Right_Thalamus"
]
missing_cov_cols = [c for c in required_cov_cols if c not in cov_df.columns]
if missing_cov_cols:
    raise ValueError(f"Missing required columns in covariate file: {missing_cov_cols}")

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

# optional: bilateral raw ROI volumes if you want to save them
cov_sub["Bilateral_Hippocampus"] = (
    pd.to_numeric(cov_sub["Left_Hippocampus"], errors="coerce") +
    pd.to_numeric(cov_sub["Right_Hippocampus"], errors="coerce")
)
cov_sub["Bilateral_Thalamus"] = (
    pd.to_numeric(cov_sub["Left_Thalamus"], errors="coerce") +
    pd.to_numeric(cov_sub["Right_Thalamus"], errors="coerce")
)

# =========================================================
# MERGE PC1 WITH COVARIATES
# =========================================================
model_df = pc1_long_df.merge(
    cov_sub,
    on=["subject_id", "session"],
    how="left"
)

print("Merged model df:", model_df.shape)
print(model_df.head())

# =========================================================
# FIT MIXED MODELS BY BILATERAL STRUCTURE
# =========================================================
results = []

for structure in target_groups.keys():
    sub = model_df[model_df["structure"] == structure].copy()
    sub = sub.dropna(subset=["PC1", "session", "sex", "BrainSegVol", "subject_id"]).copy()

    if sub["subject_id"].nunique() < 10:
        print(f"Skipping {structure}: too few subjects.")
        continue

    formula = "PC1 ~ C(session) + sex + BrainSegVol"

    try:
        model = smf.mixedlm(
            formula=formula,
            data=sub,
            groups=sub["subject_id"]
        )
        fit = model.fit(reml=False, method="lbfgs", maxiter=200)

        with open(os.path.join(output_dir, f"{structure}_mixedlm_summary.txt"), "w") as f:
            f.write(str(fit.summary()))

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
            "structure": structure,
            "n_rows": len(sub),
            "n_subjects": sub["subject_id"].nunique(),
            "estimate_session3_minus_session1": est,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "t_value": tval,
            "p_value": pval,
        })

        print(f"\n================ {structure} ================\n")
        print(fit.summary())

    except Exception as e:
        print(f"Model failed for {structure}: {e}")
        results.append({
            "structure": structure,
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
results_df.to_csv(os.path.join(output_dir, "shape_pc1_mixedlm_results_bilateral.csv"), index=False)

print("\nSaved bilateral mixed model results to:")
print(os.path.join(output_dir, "shape_pc1_mixedlm_results_bilateral.csv"))
print(results_df)