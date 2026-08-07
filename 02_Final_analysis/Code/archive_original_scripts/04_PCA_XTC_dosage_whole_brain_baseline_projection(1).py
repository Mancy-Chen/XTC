# Mingshi Chen
# Whole-brain baseline-only PCA + follow-up projection + Linear Mixed Model with XTC dosage
#
# Purpose
# -------
# This script avoids temporal/post-exposure information leakage by:
#   1) fitting the StandardScaler on baseline/pre-exposure data only;
#   2) fitting PCA on baseline/pre-exposure data only;
#   3) projecting follow-up/post-exposure data into the same baseline-defined PCA space.
#
# Main model:
#   PC1 ~ time * dose_xtc + age + sex + BrainSegVol + (1 | subject_id)
#
# Inputs:
#   1) whole_brain_volume_pre_delta.csv
#   2) xtc_dose_wide_mapped_to_y.csv
#
# Notes:
#   - Session 3 is reconstructed as post = pre + delta.
#   - BrainSegVol-like features are excluded from PCA and used as covariates.
#   - PCA feature variance filtering is based on baseline only.
#   - PCA features with missing values at either time point are dropped before scoring.
#   - Age is mean-imputed, following the previous script logic.

import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# =========================================================
# SETTINGS
# =========================================================
x_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/whole_brain_volume_pre_delta.csv"
dose_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/xtc_dose_wide_mapped_to_y.csv"

# New output folder, so results do not overwrite the old stacked-PCA analysis.
output_root = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_lmm_with_xtc_dose_baseline_projection"
os.makedirs(output_root, exist_ok=True)

subject_col = "subject_id"

# Choose the XTC dosage variable.
# Examples: "dose_post", "dose_sum_1_2_3", "dose_change_1_to_3", "xlttot_sessie3"
dose_col = "xlttot_sessie3"
use_log_dose = True

# Session labels used in the output files
pre_session_label = "sessie1"
post_session_label = "sessie3"

# BrainSegVol is used as a covariate, so exclude it from PCA features.
brainsegvol_patterns = [
    "brainsegvol",
    "brainsegvolnotvent",
]

# Optional: features with near-zero variance at baseline can destabilize PCA.
variance_threshold = 1e-12

# Number of PCs to save. PC1 is used as the primary outcome.
n_components_to_save = 5

# Drop PCA features with any missing values across retained baseline/follow-up rows.
# This preserves a rectangular feature matrix for PCA projection.
missing_feature_drop_threshold = 0.0

# =========================================================
# HELPERS
# =========================================================
def safe_numeric(series):
    """Convert strings like '1,23' to numeric 1.23."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )


def add_subject_id_from_cohort_id(df):
    """Create subject_id like I012 from cohort/id when needed."""
    if {"cohort", "id"}.issubset(df.columns) and "subject_id" not in df.columns:
        cohort_map = {1: "I", 2: "II", 3: "III", 4: "IV"}
        out = df.copy()
        out["cohort"] = pd.to_numeric(out["cohort"], errors="coerce")
        out["id"] = pd.to_numeric(out["id"], errors="coerce")
        out["subject_id"] = out.apply(
            lambda r: f"{cohort_map.get(int(r['cohort']), str(int(r['cohort'])))}{int(r['id']):03d}"
            if pd.notna(r["cohort"]) and pd.notna(r["id"]) else np.nan,
            axis=1
        )
        return out
    return df.copy()


def add_cohort_id_from_subject_id(df):
    """Parse cohort/id from subject_id like I012 when needed."""
    if "subject_id" not in df.columns:
        return df.copy()

    out = df.copy()

    def parse_subject_id(s):
        if pd.isna(s):
            return np.nan, np.nan
        s = str(s).strip()
        m = re.match(r"^(I{1,3}|IV)(\d+)$", s)
        if not m:
            return np.nan, np.nan
        cohort_roman = m.group(1)
        num = int(m.group(2))
        cohort_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
        return cohort_map.get(cohort_roman, np.nan), num

    parsed = out["subject_id"].apply(parse_subject_id)
    out["cohort"] = [x[0] for x in parsed]
    out["id"] = [x[1] for x in parsed]
    return out


def prepare_numeric_columns(df, exclude=None):
    """Convert object columns to numeric where possible."""
    exclude = set() if exclude is None else set(exclude)
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == object:
            out[c] = safe_numeric(out[c])
    return out


def pick_feature_bases(df):
    """Find feature names that have both *_pre and *_delta columns."""
    all_cols = df.columns.tolist()
    pre_cols = [c for c in all_cols if c.endswith("_pre")]
    delta_cols = [c for c in all_cols if c.endswith("_delta")]

    pre_bases = {c[:-4] for c in pre_cols}
    delta_bases = {c[:-6] for c in delta_cols}
    common_bases = sorted(pre_bases & delta_bases)

    exclude_exact = {
        "cohort", "id", "age", "sex", "session", "subject_id", "subject_code",
        dose_col, "vwrec"
    }
    common_bases = [b for b in common_bases if b not in exclude_exact]
    return common_bases


def is_brainsegvol_feature(base):
    s = base.lower()
    return any(p in s for p in brainsegvol_patterns)


def fit_mixedlm_with_fallback(formula, data, group_col):
    """Fit random-intercept LMM with optimizer fallback."""
    try:
        model = smf.mixedlm(
            formula,
            data=data,
            groups=data[group_col],
            re_formula="1"
        )
        result = model.fit(reml=False, method="lbfgs")
    except Exception as e1:
        print("First LMM fit failed; retrying with default optimizer.")
        print("First error:", repr(e1))
        model = smf.mixedlm(
            formula,
            data=data,
            groups=data[group_col]
        )
        result = model.fit(reml=False)
    return result


def save_model_outputs(result, output_dir, prefix):
    """Save statsmodels summary and coefficient table."""
    summary_path = os.path.join(output_dir, f"{prefix}_summary.txt")
    coef_path = os.path.join(output_dir, f"{prefix}_coefficients.csv")

    with open(summary_path, "w") as f:
        f.write(result.summary().as_text())

    coef_df = pd.DataFrame({
        "term": result.params.index,
        "beta": result.params.values,
        "p_value": result.pvalues.reindex(result.params.index).values,
        "std_error": result.bse.reindex(result.params.index).values,
        "z_value": result.tvalues.reindex(result.params.index).values,
    })
    coef_df.to_csv(coef_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {coef_path}")
    return coef_df


# =========================================================
# LOAD DATA
# =========================================================
X_df = pd.read_csv(x_path)
dose_df = pd.read_csv(dose_path)

X_df.columns = [c.strip() for c in X_df.columns]
dose_df.columns = [c.strip() for c in dose_df.columns]

X_df = add_subject_id_from_cohort_id(X_df)
X_df = add_cohort_id_from_subject_id(X_df)

dose_df = add_subject_id_from_cohort_id(dose_df)
dose_df = add_cohort_id_from_subject_id(dose_df)

X_df = prepare_numeric_columns(X_df, exclude={subject_col})
dose_df = prepare_numeric_columns(dose_df, exclude={subject_col, "subject_code"})

print("Loaded X:", X_df.shape)
print("Loaded dose file:", dose_df.shape)
print("Dose columns:", dose_df.columns.tolist())

if dose_col not in dose_df.columns:
    raise ValueError(
        f"'{dose_col}' not found in dose file. Available columns:\n{dose_df.columns.tolist()}"
    )

# =========================================================
# MERGE DOSAGE
# =========================================================
# The previous script merged X subject_id with dose subject_code.
# Keep that behavior, but also allow subject_id as a fallback.
id_candidates = ["subject_code", "studnr_key", subject_col]
id_col_in_dose = next((c for c in id_candidates if c in dose_df.columns), None)
if id_col_in_dose is None:
    raise ValueError(
        "Could not find a subject ID column in dose file. "
        "Expected one of: subject_code, studnr_key, subject_id."
    )

dose_keep_cols = [c for c in [id_col_in_dose, "cohort_x", "vwrec", dose_col] if c in dose_df.columns]
dose_small = dose_df[dose_keep_cols].copy()
dose_small[id_col_in_dose] = dose_small[id_col_in_dose].astype(str).str.strip()
dose_small = dose_small.rename(columns={id_col_in_dose: "dose_subject_id"})

X_df[subject_col] = X_df[subject_col].astype(str).str.strip()

print(f"Merging X {subject_col} with dose ID column: {id_col_in_dose}")
df = X_df.merge(
    dose_small,
    left_on=subject_col,
    right_on="dose_subject_id",
    how="left"
)

print("Merged df:", df.shape)
print(f"Missing {dose_col} rows after merge:", df[dose_col].isna().sum())

# =========================================================
# RECONSTRUCT POST = PRE + DELTA
# =========================================================
feature_bases = pick_feature_bases(df)
print("Common pre/delta bases:", len(feature_bases))

for base in feature_bases:
    pre_col = f"{base}_pre"
    delta_col = f"{base}_delta"
    post_col = f"{base}_post"

    df[pre_col] = pd.to_numeric(df[pre_col], errors="coerce")
    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
    df[post_col] = df[pre_col] + df[delta_col]

# =========================================================
# FIND BrainSegVol COVARIATE
# =========================================================
brainsegvol_base_candidates = [b for b in feature_bases if is_brainsegvol_feature(b)]
if len(brainsegvol_base_candidates) == 0:
    raise ValueError("No BrainSegVol-like feature found.")

# Prefer exact BrainSegVol before BrainSegVolNotVent if both exist.
brainsegvol_base_candidates = sorted(
    brainsegvol_base_candidates,
    key=lambda x: (0 if x.lower().endswith("brainsegvol") else 1, x.lower())
)
brainsegvol_base = brainsegvol_base_candidates[0]
brainsegvol_pre_col = f"{brainsegvol_base}_pre"
brainsegvol_post_col = f"{brainsegvol_base}_post"
brainsegvol_delta_col = f"{brainsegvol_base}_delta"

print("Using BrainSegVol covariate:", brainsegvol_base)

# =========================================================
# PREPARE PCA FEATURES
# =========================================================
pca_feature_bases = [b for b in feature_bases if not is_brainsegvol_feature(b)]
print("PCA feature bases after excluding BrainSegVol-like features:", len(pca_feature_bases))

# Baseline-only variance filter. This avoids using follow-up values to define the PCA feature space.
valid_feature_names = []
for base in pca_feature_bases:
    pre_col = f"{base}_pre"
    s = pd.to_numeric(df[pre_col], errors="coerce")
    if s.notna().sum() >= 3:
        v = np.nanvar(s.to_numpy(dtype=float))
        if np.isfinite(v) and v > variance_threshold:
            valid_feature_names.append(base)

pca_feature_bases = [b for b in pca_feature_bases if b in valid_feature_names]
print("PCA feature bases after baseline-only variance filter:", len(pca_feature_bases))

if len(pca_feature_bases) < 2:
    raise ValueError("Not enough baseline-variable whole-brain features for PCA after filtering.")

# =========================================================
# BUILD LONG DATA
# =========================================================
meta_cols = [subject_col]
for col in ["cohort", "id", "sex", "age", dose_col, "vwrec"]:
    if col in df.columns and col not in meta_cols:
        meta_cols.append(col)

pre_feature_cols = [f"{b}_pre" for b in pca_feature_bases]
post_feature_cols = [f"{b}_post" for b in pca_feature_bases]

pre_long = df[meta_cols + [brainsegvol_pre_col] + pre_feature_cols].copy()
pre_long["time"] = 0
pre_long["session"] = pre_session_label
pre_long = pre_long.rename(columns={brainsegvol_pre_col: "BrainSegVol"})
pre_long = pre_long.rename(columns={f"{b}_pre": b for b in pca_feature_bases})

post_long = df[meta_cols + [brainsegvol_post_col] + post_feature_cols].copy()
post_long["time"] = 1
post_long["session"] = post_session_label
post_long = post_long.rename(columns={brainsegvol_post_col: "BrainSegVol"})
post_long = post_long.rename(columns={f"{b}_post": b for b in pca_feature_bases})

long_df = pd.concat([pre_long, post_long], axis=0, ignore_index=True)

# =========================================================
# IMPUTE AGE
# =========================================================
if "age" in long_df.columns:
    long_df["age"] = pd.to_numeric(long_df["age"], errors="coerce")
    n_missing_age_before = long_df["age"].isna().sum()
    mean_age = long_df["age"].mean(skipna=True)
    long_df["age"] = long_df["age"].fillna(mean_age)
    n_missing_age_after = long_df["age"].isna().sum()
    print(f"Imputed missing age with mean age = {mean_age:.3f}")
    print(f"Missing age before imputation: {n_missing_age_before}")
    print(f"Missing age after imputation: {n_missing_age_after}")

# =========================================================
# DIAGNOSTICS BEFORE COMPLETE-CASE FILTER
# =========================================================
print("\n===== DIAGNOSTICS BEFORE COMPLETE-CASE FILTER =====")
print("Rows in long_df:", long_df.shape[0])
print("Unique subjects in long_df:", long_df[subject_col].nunique())
for c in ["age", "sex", "BrainSegVol", dose_col]:
    if c in long_df.columns:
        print(f"Missing {c}: {long_df[c].isna().sum()}")

feature_missing_counts = long_df[pca_feature_bases].isna().sum().sort_values(ascending=False)
print("\nTop 20 missing PCA features:")
print(feature_missing_counts.head(20))

# Drop PCA features with any missingness across baseline/follow-up.
# This is not used to learn PCA directions; it only ensures projectable matrices.
missing_prop = long_df[pca_feature_bases].isna().mean()
bad_features = missing_prop[missing_prop > missing_feature_drop_threshold].index.tolist()
print(f"\nDropping {len(bad_features)} PCA features with > {missing_feature_drop_threshold*100:.1f}% missingness")
print(bad_features)

pca_feature_bases = [f for f in pca_feature_bases if f not in bad_features]
print("Remaining PCA features:", len(pca_feature_bases))

if len(pca_feature_bases) < 2:
    raise ValueError("Not enough PCA features remaining after missingness filtering.")

# =========================================================
# COMPLETE-CASE FILTER AND PAIRED SUBJECTS
# =========================================================
needed_cols = [subject_col, "time", "session", "BrainSegVol", "age", "sex", dose_col] + pca_feature_bases
needed_cols = [c for c in needed_cols if c in long_df.columns]

long_complete = long_df.dropna(subset=needed_cols).copy()

print("\nLong complete data before paired filter:", long_complete.shape)
print("Unique subjects before paired filter:", long_complete[subject_col].nunique())

pair_counts = long_complete.groupby(subject_col)["time"].nunique()
paired_subjects = pair_counts[pair_counts == 2].index.tolist()
print("Paired subjects retained:", len(paired_subjects))

long_complete = long_complete[long_complete[subject_col].isin(paired_subjects)].copy()

if len(paired_subjects) < 3:
    raise ValueError("Too few paired subjects for mixed model.")

# Sort to guarantee baseline/follow-up rows align by subject.
baseline_df = (
    long_complete[long_complete["time"] == 0]
    .sort_values(subject_col)
    .reset_index(drop=True)
)
followup_df = (
    long_complete[long_complete["time"] == 1]
    .sort_values(subject_col)
    .reset_index(drop=True)
)

if baseline_df[subject_col].tolist() != followup_df[subject_col].tolist():
    raise ValueError("Baseline and follow-up subject order does not match after sorting.")

# =========================================================
# BASELINE-ONLY PCA + FOLLOW-UP PROJECTION
# =========================================================
# Important anti-leakage step:
#   scaler.fit and pca.fit are performed only on baseline/pre-exposure data.
#   follow-up/post-exposure data are only transformed/projected.
X_base = baseline_df[pca_feature_bases].to_numpy(dtype=float)
X_follow = followup_df[pca_feature_bases].to_numpy(dtype=float)

scaler = StandardScaler()
X_base_scaled = scaler.fit_transform(X_base)
X_follow_scaled = scaler.transform(X_follow)

n_components = min(n_components_to_save, X_base_scaled.shape[1], X_base_scaled.shape[0])
pca = PCA(n_components=n_components, random_state=42)
base_scores = pca.fit_transform(X_base_scaled)
follow_scores = pca.transform(X_follow_scaled)

pc_cols = [f"PC{i+1}" for i in range(n_components)]

base_scores_df = pd.DataFrame(base_scores, columns=pc_cols)
follow_scores_df = pd.DataFrame(follow_scores, columns=pc_cols)

baseline_scored = pd.concat([baseline_df.reset_index(drop=True), base_scores_df], axis=1)
followup_scored = pd.concat([followup_df.reset_index(drop=True), follow_scores_df], axis=1)

long_scored = pd.concat([baseline_scored, followup_scored], axis=0, ignore_index=True)
long_scored = long_scored.sort_values([subject_col, "time"]).reset_index(drop=True)

# Loadings and explained variance are baseline-defined.
loadings = pd.DataFrame(
    pca.components_.T,
    index=pca_feature_bases,
    columns=pc_cols
).reset_index().rename(columns={"index": "feature"})

explained_var_df = pd.DataFrame({
    "component": pc_cols,
    "explained_variance_ratio_baseline_fit": pca.explained_variance_ratio_,
    "cumulative_explained_variance_ratio_baseline_fit": np.cumsum(pca.explained_variance_ratio_)
})

# Save scaling information so you can document/reproduce projection.
scaler_params = pd.DataFrame({
    "feature": pca_feature_bases,
    "baseline_scaler_mean": scaler.mean_,
    "baseline_scaler_scale": scaler.scale_,
})

pca_config = {
    "pca_strategy": "baseline_only_fit_followup_projection",
    "scaler_fit": "baseline/pre-exposure observations only",
    "pca_fit": "baseline/pre-exposure observations only",
    "followup_scores": "computed using scaler.transform and pca.transform",
    "n_subjects_paired": int(len(paired_subjects)),
    "n_features_pca": int(len(pca_feature_bases)),
    "n_components_saved": int(n_components),
    "dose_col": dose_col,
    "use_log_dose": bool(use_log_dose),
    "brainsegvol_covariate": brainsegvol_base,
}

# =========================================================
# SAVE PCA OUTPUTS
# =========================================================
long_scores_path = os.path.join(output_root, "whole_brain_long_with_pca_scores_and_dose_baseline_projection.csv")
loadings_path = os.path.join(output_root, "whole_brain_pca_loadings_baseline_fit.csv")
explained_path = os.path.join(output_root, "whole_brain_pca_explained_variance_baseline_fit.csv")
scaler_path = os.path.join(output_root, "whole_brain_pca_baseline_scaler_parameters.csv")
config_path = os.path.join(output_root, "whole_brain_pca_baseline_projection_config.json")

long_scored.to_csv(long_scores_path, index=False)
loadings.to_csv(loadings_path, index=False)
explained_var_df.to_csv(explained_path, index=False)
scaler_params.to_csv(scaler_path, index=False)
with open(config_path, "w") as f:
    json.dump(pca_config, f, indent=2)

# Wide PC1 file for subject-level change.
wide_index_cols = [c for c in [subject_col, "cohort", "id", "age", "sex", dose_col, "vwrec"] if c in long_scored.columns]
pc1_wide = long_scored.pivot_table(
    index=wide_index_cols,
    columns="session",
    values="PC1"
).reset_index()

if {pre_session_label, post_session_label}.issubset(pc1_wide.columns):
    pc1_wide["PC1_delta"] = pc1_wide[post_session_label] - pc1_wide[pre_session_label]
else:
    pc1_wide["PC1_delta"] = np.nan

# Add BrainSegVol pre/post/delta to the wide file.
brainsegvol_delta_df = df[[subject_col, brainsegvol_pre_col, brainsegvol_post_col, brainsegvol_delta_col]].copy()
brainsegvol_delta_df = brainsegvol_delta_df.rename(columns={
    brainsegvol_pre_col: "BrainSegVol_pre",
    brainsegvol_post_col: "BrainSegVol_post",
    brainsegvol_delta_col: "BrainSegVol_delta",
})
pc1_wide = pc1_wide.merge(brainsegvol_delta_df, on=subject_col, how="left")
pc1_wide.to_csv(os.path.join(output_root, "whole_brain_pc1_subject_wide_with_dose_baseline_projection.csv"), index=False)

# Save missingness diagnostics.
feature_missing_counts.to_csv(
    os.path.join(output_root, "whole_brain_feature_missing_counts_long_baseline_projection.csv"),
    header=["n_missing"]
)

subject_pair_counts = long_scored.groupby(subject_col)["time"].nunique().value_counts().sort_index()
subject_pair_counts.to_csv(
    os.path.join(output_root, "whole_brain_subject_pair_counts_baseline_projection.csv"),
    header=["n_subjects"]
)

# =========================================================
# PREPARE DOSE AND COVARIATES FOR MODELS
# =========================================================
long_scored[dose_col] = pd.to_numeric(long_scored[dose_col], errors="coerce")

if (long_scored[dose_col].dropna() < 0).any():
    raise ValueError(f"{dose_col} contains negative values; log1p dose is not appropriate.")

if use_log_dose:
    long_scored["dose_xtc"] = np.log1p(long_scored[dose_col])
    print(f"Using log-transformed dose: log1p({dose_col})")
else:
    long_scored["dose_xtc"] = long_scored[dose_col]
    print(f"Using raw dose: {dose_col}")

# Center continuous covariates for model stability.
long_scored["age_c"] = long_scored["age"] - long_scored["age"].mean()
long_scored["BrainSegVol_c"] = long_scored["BrainSegVol"] - long_scored["BrainSegVol"].mean()
long_scored["dose_xtc_c"] = long_scored["dose_xtc"] - long_scored["dose_xtc"].mean()
long_scored["sex"] = long_scored["sex"].astype("category")

# =========================================================
# LMM 1: BASIC LONGITUDINAL PC1 MODEL
# =========================================================
basic_lmm_df = long_scored[
    [subject_col, "PC1", "time", "age_c", "sex", "BrainSegVol_c"]
].dropna().copy()

basic_formula = "PC1 ~ time + age_c + C(sex) + BrainSegVol_c"
basic_res = fit_mixedlm_with_fallback(basic_formula, basic_lmm_df, subject_col)
basic_params = save_model_outputs(
    basic_res,
    output_root,
    "whole_brain_pc1_lmm_basic_baseline_projection"
)

# =========================================================
# LMM 2: PRIMARY XTC DOSE INTERACTION MODEL
# =========================================================
dose_lmm_df = long_scored[
    [subject_col, "PC1", "time", "dose_xtc_c", "age_c", "sex", "BrainSegVol_c"]
].dropna().copy()

primary_formula = "PC1 ~ time * dose_xtc_c + age_c + C(sex) + BrainSegVol_c"
primary_res = fit_mixedlm_with_fallback(primary_formula, dose_lmm_df, subject_col)
primary_params = save_model_outputs(
    primary_res,
    output_root,
    "whole_brain_pc1_lmm_with_dose_baseline_projection"
)

# =========================================================
# OPTIONAL: EXPLORATORY PC2-PC5 DOSE MODELS
# =========================================================
exploratory_rows = []
for pc in pc_cols[1:]:
    tmp_df = long_scored[
        [subject_col, pc, "time", "dose_xtc_c", "age_c", "sex", "BrainSegVol_c"]
    ].dropna().copy()
    if tmp_df.shape[0] < 6 or tmp_df[subject_col].nunique() < 3:
        continue

    formula = f"{pc} ~ time * dose_xtc_c + age_c + C(sex) + BrainSegVol_c"
    try:
        res = fit_mixedlm_with_fallback(formula, tmp_df, subject_col)
        for term in res.params.index:
            exploratory_rows.append({
                "component": pc,
                "term": term,
                "beta": res.params[term],
                "p_value": res.pvalues.get(term, np.nan),
                "std_error": res.bse.get(term, np.nan),
                "z_value": res.tvalues.get(term, np.nan),
            })
    except Exception as e:
        exploratory_rows.append({
            "component": pc,
            "term": "MODEL_FAILED",
            "beta": np.nan,
            "p_value": np.nan,
            "std_error": np.nan,
            "z_value": np.nan,
            "error": repr(e),
        })

exploratory_df = pd.DataFrame(exploratory_rows)
exploratory_df.to_csv(
    os.path.join(output_root, "whole_brain_pc2_pc5_lmm_with_dose_baseline_projection.csv"),
    index=False
)

# =========================================================
# PRINT SUMMARY
# =========================================================
print("\nFinished baseline-only whole-brain PCA + follow-up projection + LMM with XTC dose.")
print("Saved to:", output_root)

print("\nAnti-leakage check:")
print("- StandardScaler was fitted on baseline only.")
print("- PCA was fitted on baseline only.")
print("- Follow-up scores were computed with scaler.transform and pca.transform.")

print("\nPCA explained variance from baseline fit:")
print(explained_var_df.to_string(index=False))

print("\nBasic LMM coefficients:")
print(basic_params.to_string(index=False))

print("\nPrimary dose-interaction LMM coefficients:")
print(primary_params.to_string(index=False))

print("\nKey interpretation guide:")
print("- time                 = overall Session 1 -> Session 3 shift in baseline-defined PC1")
print("- dose_xtc_c           = baseline PC1 difference by future/follow-up cumulative dose")
print("- time:dose_xtc_c      = whether higher dose predicts larger longitudinal PC1 change")
print("- PC1 sign is arbitrary; interpret the time*dose effect relative to the reported loadings.")
