# Mancy Chen
# Baseline-fit PCA for radiomics voxel-volume data
#
# Core principle:
#   1. Fit the StandardScaler using baseline ROI volumes only.
#   2. Fit PCA using standardized baseline ROI volumes only.
#   3. Reconstruct follow-up ROI volumes as post = pre + delta.
#   4. Project both baseline and follow-up through the SAME baseline scaler
#      and PCA model.
#   5. Compute PC delta = PC_post - PC_pre.
#
# This prevents follow-up information from influencing the PCA axes.

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================================================
# SETTINGS
# =========================================================
INPUT_CSV = Path(
    "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/whole_brain_volume/"
    "New_radiomics_voxel_volume/radiomics_voxelvolume_pre_delta_95_subjects_cm3_with_xtc_ravlt_covariates.csv"
)

OUTPUT_DIR = Path(
    "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_radiomics_voxel_volume_baseline_projection"
)

SUBJECT_COL = "subject_id"
BRAINSEG_COL = "aseg+DKT_BrainSegVol_pre"

# Export the first N PCs for downstream analyses.
# The PCA model itself is fitted with the maximum estimable number of PCs.
N_COMPONENTS_TO_EXPORT = 5

# Remove features with essentially zero variance at baseline.
BASELINE_VARIANCE_THRESHOLD = 1e-12

# PCA signs are mathematically arbitrary. When True, each component is
# oriented so that the sum of its ROI weights is positive. This usually
# makes a global-volume PC easier to interpret.
ORIENT_PC_SIGNS = True


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def clean_numeric(series: pd.Series, column_name: str) -> pd.Series:
    """Convert one column to numeric and report invalid values clearly."""
    converted = pd.to_numeric(series, errors="coerce")

    newly_missing = converted.isna() & series.notna()
    if newly_missing.any():
        bad_examples = series.loc[newly_missing].astype(str).unique()[:5]
        raise ValueError(
            f"Column '{column_name}' contains non-numeric values. "
            f"Examples: {bad_examples.tolist()}"
        )

    return converted


def identify_roi_pairs(df: pd.DataFrame):
    """
    Identify radiomics ROI pre/delta pairs.

    Explicitly excludes behavioral RAVLT columns and BrainSegVol because
    they are covariates, not ROI features for PCA.
    """
    excluded_pre_columns = {
        "vwrec_pre",
        BRAINSEG_COL,
    }

    pre_columns = [
        column
        for column in df.columns
        if column.endswith("_pre")
        and column not in excluded_pre_columns
    ]

    roi_names = [column[:-4] for column in pre_columns]
    delta_columns = [f"{roi}_delta" for roi in roi_names]

    missing_delta_columns = [
        delta_column
        for delta_column in delta_columns
        if delta_column not in df.columns
    ]

    if missing_delta_columns:
        raise ValueError(
            "The following ROI delta columns are missing:\n"
            + "\n".join(missing_delta_columns)
        )

    if not pre_columns:
        raise ValueError("No paired ROI pre/delta columns were detected.")

    return roi_names, pre_columns, delta_columns


# =========================================================
# LOAD DATA
# =========================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

if SUBJECT_COL not in df.columns:
    raise KeyError(f"Missing subject identifier column: {SUBJECT_COL}")

if df[SUBJECT_COL].duplicated().any():
    duplicates = (
        df.loc[df[SUBJECT_COL].duplicated(), SUBJECT_COL]
        .astype(str)
        .tolist()
    )
    raise ValueError(f"Duplicate subjects found: {duplicates}")

roi_names, pre_columns, delta_columns = identify_roi_pairs(df)

print(f"Subjects: {len(df)}")
print(f"Paired ROI features detected: {len(roi_names)}")


# =========================================================
# CONSTRUCT BASELINE, FOLLOW-UP, AND DELTA MATRICES
# =========================================================
X_pre = pd.DataFrame(index=df.index)
X_delta = pd.DataFrame(index=df.index)

for roi, pre_column, delta_column in zip(
    roi_names,
    pre_columns,
    delta_columns,
):
    X_pre[roi] = clean_numeric(df[pre_column], pre_column)
    X_delta[roi] = clean_numeric(df[delta_column], delta_column)

if X_pre.isna().any().any():
    missing = X_pre.isna().sum()
    missing = missing[missing > 0]
    raise ValueError(
        "Missing baseline ROI values:\n"
        + missing.to_string()
    )

if X_delta.isna().any().any():
    missing = X_delta.isna().sum()
    missing = missing[missing > 0]
    raise ValueError(
        "Missing ROI delta values:\n"
        + missing.to_string()
    )

# Follow-up volume is baseline volume plus observed change.
X_post = X_pre + X_delta


# =========================================================
# BASELINE-ONLY VARIANCE FILTER
# =========================================================
# The variance decision is made from baseline data only.
baseline_variance = X_pre.var(axis=0, ddof=0)

kept_roi_names = baseline_variance[
    baseline_variance > BASELINE_VARIANCE_THRESHOLD
].index.tolist()

removed_roi_names = baseline_variance[
    baseline_variance <= BASELINE_VARIANCE_THRESHOLD
].index.tolist()

if not kept_roi_names:
    raise ValueError("No ROI features remained after variance filtering.")

X_pre_kept = X_pre[kept_roi_names]
X_post_kept = X_post[kept_roi_names]

print(f"ROIs retained after baseline variance filtering: {len(kept_roi_names)}")
print(f"ROIs removed for zero/near-zero baseline variance: {len(removed_roi_names)}")

if removed_roi_names:
    print("Removed ROIs:")
    for roi in removed_roi_names:
        print(f"  - {roi}")


# =========================================================
# FIT SCALER ON BASELINE ONLY
# =========================================================
scaler = StandardScaler(with_mean=True, with_std=True)

Z_pre = scaler.fit_transform(X_pre_kept)
Z_post = scaler.transform(X_post_kept)

# Important:
# scaler.fit_transform() is used only for baseline.
# Follow-up uses scaler.transform(), so baseline means and SDs are reused.


# =========================================================
# FIT PCA ON BASELINE ONLY
# =========================================================
# Centered data from N subjects has rank at most N - 1.
maximum_components = min(
    len(df) - 1,
    len(kept_roi_names),
)

pca = PCA(
    n_components=maximum_components,
    svd_solver="full",
)

PC_pre = pca.fit_transform(Z_pre)
PC_post = pca.transform(Z_post)
PC_delta = PC_post - PC_pre

# Important:
# pca.fit_transform() is used only for baseline.
# Follow-up uses pca.transform(), so it cannot influence the PCA axes.


# =========================================================
# OPTIONAL DETERMINISTIC SIGN ORIENTATION
# =========================================================
# Flipping a PCA component and all its scores does not change the model.
# This step only makes the reported direction more intuitive/reproducible.
component_signs = np.ones(pca.n_components_, dtype=float)

if ORIENT_PC_SIGNS:
    component_sums = pca.components_.sum(axis=1)
    component_signs[component_sums < 0] = -1.0

PC_pre = PC_pre * component_signs
PC_post = PC_post * component_signs
PC_delta = PC_delta * component_signs
oriented_components = pca.components_ * component_signs[:, np.newaxis]


# =========================================================
# NUMBER OF PCs TO EXPORT
# =========================================================
n_export = min(
    N_COMPONENTS_TO_EXPORT,
    pca.n_components_,
)

pc_names = [f"PC{i}" for i in range(1, n_export + 1)]

print(f"Maximum PCA components fitted: {pca.n_components_}")
print(f"PCs exported for downstream analysis: {n_export}")


# =========================================================
# CREATE WIDE SUBJECT-LEVEL SCORE FILE
# =========================================================
# Keep all original non-ROI variables, including XTC dose, RAVLT,
# polysubstance variables, and baseline BrainSegVol.
roi_columns_to_remove = set(pre_columns + delta_columns)
metadata_columns = [
    column
    for column in df.columns
    if column not in roi_columns_to_remove
]

scores_wide = df[metadata_columns].copy()

for pc_index, pc_name in enumerate(pc_names):
    scores_wide[f"{pc_name}_pre"] = PC_pre[:, pc_index]
    scores_wide[f"{pc_name}_post"] = PC_post[:, pc_index]
    scores_wide[f"{pc_name}_delta"] = PC_delta[:, pc_index]

wide_path = OUTPUT_DIR / "radiomics_pca_scores_baseline_fit_wide.csv"
scores_wide.to_csv(wide_path, index=False)


# =========================================================
# CREATE LONG SCORE FILE FOR LMM
# =========================================================
# Baseline covariates are repeated across the two time rows.
metadata = df[metadata_columns].copy()

pre_long = metadata.copy()
pre_long["time"] = 0
pre_long["session"] = "pre"

post_long = metadata.copy()
post_long["time"] = 1
post_long["session"] = "post"

for pc_index, pc_name in enumerate(pc_names):
    pre_long[pc_name] = PC_pre[:, pc_index]
    post_long[pc_name] = PC_post[:, pc_index]

scores_long = pd.concat(
    [pre_long, post_long],
    axis=0,
    ignore_index=True,
)

scores_long = scores_long.sort_values(
    by=[SUBJECT_COL, "time"],
    kind="stable",
).reset_index(drop=True)

long_path = OUTPUT_DIR / "radiomics_pca_scores_baseline_fit_long.csv"
scores_long.to_csv(long_path, index=False)


# =========================================================
# SAVE PCA LOADINGS
# =========================================================
# Here "loading" means the PCA component weight from sklearn.
loadings = pd.DataFrame({
    "roi": kept_roi_names,
    "baseline_mean_cm3": scaler.mean_,
    "baseline_sd_cm3": scaler.scale_,
    "baseline_variance_cm6": baseline_variance.loc[
        kept_roi_names
    ].to_numpy(),
})

for pc_index, pc_name in enumerate(pc_names):
    loadings[f"{pc_name}_loading"] = oriented_components[
        pc_index, :
    ]

loadings_path = OUTPUT_DIR / "radiomics_pca_loadings_baseline_fit.csv"
loadings.to_csv(loadings_path, index=False)


# =========================================================
# SAVE EXPLAINED VARIANCE
# =========================================================
explained_variance = pd.DataFrame({
    "component": [
        f"PC{i}"
        for i in range(1, pca.n_components_ + 1)
    ],
    "explained_variance": pca.explained_variance_,
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance_ratio": np.cumsum(
        pca.explained_variance_ratio_
    ),
    "orientation_sign": component_signs,
})

variance_path = (
    OUTPUT_DIR
    / "radiomics_pca_explained_variance_baseline_fit.csv"
)
explained_variance.to_csv(variance_path, index=False)


# =========================================================
# SAVE FEATURE AUDIT
# =========================================================
feature_audit = pd.DataFrame({
    "roi": roi_names,
    "baseline_variance_cm6": baseline_variance.loc[
        roi_names
    ].to_numpy(),
    "retained_for_pca": [
        roi in kept_roi_names
        for roi in roi_names
    ],
})

feature_audit_path = (
    OUTPUT_DIR
    / "radiomics_pca_feature_audit_baseline_fit.csv"
)
feature_audit.to_csv(feature_audit_path, index=False)


# =========================================================
# SAVE FITTED MODELS
# =========================================================
model_bundle = {
    "scaler": scaler,
    "pca": pca,
    "kept_roi_names": kept_roi_names,
    "removed_roi_names": removed_roi_names,
    "component_signs": component_signs,
    "n_components_exported": n_export,
    "input_units": "cm3",
    "fit_sample": "baseline only",
    "post_definition": "pre + delta",
}

model_path = OUTPUT_DIR / "radiomics_pca_baseline_fit_model.joblib"
joblib.dump(model_bundle, model_path)


# =========================================================
# VERIFICATION
# =========================================================
# Verify mathematically that score differences equal projected post-pre.
if not np.allclose(
    PC_delta,
    PC_post - PC_pre,
    rtol=0.0,
    atol=1e-12,
):
    raise AssertionError("PC delta verification failed.")

# Since the scaler was fitted on baseline, baseline standardized means
# should be approximately zero.
if not np.allclose(
    Z_pre.mean(axis=0),
    0.0,
    atol=1e-10,
):
    raise AssertionError("Baseline scaler centering verification failed.")

# Baseline PCA scores should also be centered.
if not np.allclose(
    PC_pre.mean(axis=0),
    0.0,
    atol=1e-10,
):
    raise AssertionError("Baseline PCA score centering verification failed.")


# =========================================================
# SUMMARY
# =========================================================
print("\nBaseline-fit PCA completed successfully.")
print(f"Input file: {INPUT_CSV}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Subjects: {len(df)}")
print(f"Baseline ROI features entered: {len(roi_names)}")
print(f"Baseline ROI features retained: {len(kept_roi_names)}")
print("\nExplained variance of exported PCs:")

for pc_index, pc_name in enumerate(pc_names):
    ratio = pca.explained_variance_ratio_[pc_index]
    cumulative = np.cumsum(
        pca.explained_variance_ratio_
    )[pc_index]
    print(
        f"  {pc_name}: {ratio:.4%} "
        f"(cumulative: {cumulative:.4%})"
    )

print("\nCreated files:")
for path in [
    wide_path,
    long_path,
    loadings_path,
    variance_path,
    feature_audit_path,
    model_path,
]:
    print(f"  {path}")
