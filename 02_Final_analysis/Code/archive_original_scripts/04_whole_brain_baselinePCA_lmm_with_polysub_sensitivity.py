# Mingshi Chen
# Whole-brain baseline-defined PCA + Linear Mixed Model with XTC dosage
# Updated for:
#   1) PCA fitted on baseline/pre-exposure observations only
#   2) Follow-up observations projected into the baseline-defined PCA space
#   3) Primary imaging model
#   4) Polysubstance-adjusted sensitivity model
#   5) Optional sensitivity excluding non-XTC polydrug users
#
# Primary imaging model:
#   PC1 ~ time * XTC_dose + age + sex + BrainSegVol + (1 | subject_id)
#
# Sensitivity model 1, polysubstance-adjusted:
#   PC1 ~ time * XTC_dose
#       + age + sex + BrainSegVol
#       + cannabis + alcohol + tobacco + amphetamine + cocaine
#       + (1 | subject_id)
#
# Sensitivity model 2, optional polydrug-user exclusion:
#   Same as primary model, after excluding participants with selected non-XTC drug use
#
# Inputs:
#   1) whole_brain_volume_pre_delta.csv
#   2) xtc_dose_wide_mapped_with_ravlt_selected.csv
#
# Notes:
# - Session 3 is reconstructed as post = pre + delta.
# - PCA excludes BrainSegVol features.
# - Near-zero variance filtering is based on baseline data only.
# - PCA scaling and loadings are fitted on baseline data only.
# - Follow-up data are projected using the baseline-fitted scaler and PCA loadings.
# - Age is mean-imputed.
# - Polysubstance columns are read from the dose/RAVLT/substance file.

import os
import re
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
dose_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/XTC_dosage/xtc_dose_wide_mapped_with_ravlt_selected.csv"

output_root = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_baselinePCA_lmm_with_xtc_dose"
os.makedirs(output_root, exist_ok=True)

subject_col = "subject_id"

# Choose the XTC dosage variable you want to test.
# If your final dose file uses xpillen3 instead, change this to "xpillen3".
dose_col = "xlttot_sessie3"
use_log_dose = True

# Polysubstance covariates.
# Rename here if your file uses different column names.
polysub_cols = {
    "cannabis": "lca1jt",
    "alcohol": "lalupw",
    "tobacco": "lsigpw",
    "amphetamine": "ls1jht",
    "cocaine": "lco1jt",
}

# BrainSegVol is used as a covariate, not as a PCA feature.
brainsegvol_patterns = [
    "brainsegvol",
    "brainsegvolnotvent",
]

variance_threshold = 1e-12
n_pcs_to_save = 5

# Optional sensitivity excluding non-XTC polydrug users.
# This is separate from the polysubstance-adjusted model above.
run_polydrug_exclusion_sensitivity = True

# By default, define non-XTC polydrug use using cannabis, amphetamine, and cocaine.
# Alcohol/tobacco are excluded from this default because they may remove too many subjects.
# For a stricter sensitivity, add "alcohol" and/or "tobacco".
polydrug_exclusion_clean_names = ["cannabis", "amphetamine", "cocaine"]

# Your substance variables often appear log-transformed with no-use coded as ln(0.5) = -0.693147.
# If your variables are raw counts with no-use coded 0, the helper below also handles that.
logged_no_use_code = -0.6931471805599453
polydrug_tolerance = 1e-6


# =========================================================
# HELPERS
# =========================================================
def safe_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )


def add_subject_id_from_cohort_id(df):
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
    exclude = set() if exclude is None else set(exclude)
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == object:
            out[c] = safe_numeric(out[c])
    return out


def pick_feature_bases(df):
    all_cols = df.columns.tolist()
    pre_cols = [c for c in all_cols if c.endswith("_pre")]
    delta_cols = [c for c in all_cols if c.endswith("_delta")]

    pre_bases = {c[:-4] for c in pre_cols}
    delta_bases = {c[:-6] for c in delta_cols}
    common_bases = sorted(pre_bases & delta_bases)

    exclude_exact = {
        "cohort", "id", "age", "sex", "session", "subject_id", "subject_code",
        "vwrec", dose_col,
    }
    return [b for b in common_bases if b not in exclude_exact]


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


def save_lmm_outputs(result, output_dir, prefix):
    summary_path = os.path.join(output_dir, f"{prefix}_summary.txt")
    coef_path = os.path.join(output_dir, f"{prefix}_coefficients.csv")

    with open(summary_path, "w") as f:
        f.write(result.summary().as_text())

    params = pd.DataFrame({
        "term": result.params.index,
        "beta": result.params.values,
        "p_value": result.pvalues.reindex(result.params.index).values,
        "std_error": result.bse.reindex(result.params.index).values,
        "z_value": result.tvalues.reindex(result.params.index).values,
    })
    params.to_csv(coef_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {coef_path}")
    return params


def is_nonuse_or_missing_like(values):
    """Return True where values are consistent with no-use.

    Handles two common encodings:
    - raw/count-like no-use = 0
    - log-transformed no-use = ln(0.5) = -0.693147...
    """
    s = pd.to_numeric(values, errors="coerce")
    # If the variable has negative values, assume logged no-use coding.
    if s.min(skipna=True) < 0:
        return s <= (logged_no_use_code + polydrug_tolerance)
    # Otherwise assume raw/count-like coding.
    return s <= polydrug_tolerance


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
dose_df = prepare_numeric_columns(dose_df, exclude={subject_col, "subject_code", "studnr_key"})

print("Loaded X:", X_df.shape)
print("Loaded dose file:", dose_df.shape)
print("Dose columns:", dose_df.columns.tolist())

if dose_col not in dose_df.columns:
    raise ValueError(
        f"'{dose_col}' not found in dose file. Available columns:\n{dose_df.columns.tolist()}"
    )


# =========================================================
# KEEP DOSAGE + POLYSUBSTANCE COLUMNS
# =========================================================
poly_file_cols = list(polysub_cols.values())

id_candidates = ["subject_code", "studnr_key", "subject_id"]
id_col_in_dose = next((c for c in id_candidates if c in dose_df.columns), None)
if id_col_in_dose is None:
    raise ValueError(
        "Could not find a subject ID column in dose file. "
        "Expected one of: subject_code, studnr_key, subject_id."
    )

dose_keep_cols = [
    c for c in [id_col_in_dose, "cohort_x", "vwrec", dose_col] + poly_file_cols
    if c in dose_df.columns
]
dose_small = dose_df[dose_keep_cols].copy()

missing_poly_cols = [c for c in poly_file_cols if c not in dose_df.columns]
if missing_poly_cols:
    print("WARNING: These polysubstance columns were not found in dose file:")
    print(missing_poly_cols)

dose_small[id_col_in_dose] = dose_small[id_col_in_dose].astype(str).str.strip()
dose_small = dose_small.rename(columns={id_col_in_dose: "subject_code"})

X_df[subject_col] = X_df[subject_col].astype(str).str.strip()

print(f"Merging X {subject_col} with dose subject_code, originally {id_col_in_dose}")

df = X_df.merge(
    dose_small,
    left_on=subject_col,
    right_on="subject_code",
    how="left"
)

print("Merged df:", df.shape)
print(f"Missing {dose_col} rows after merge:", df[dose_col].isna().sum())
for raw_col in poly_file_cols:
    if raw_col in df.columns:
        print(f"Missing {raw_col} rows after merge:", df[raw_col].isna().sum())


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

brainsegvol_base_candidates = sorted(
    brainsegvol_base_candidates,
    key=lambda x: (0 if x.lower().endswith("brainsegvol") else 1, x.lower())
)
brainsegvol_base = brainsegvol_base_candidates[0]

brainsegvol_pre_col = f"{brainsegvol_base}_pre"
brainsegvol_post_col = f"{brainsegvol_base}_post"

print("Using BrainSegVol covariate:", brainsegvol_base)


# =========================================================
# PREPARE PCA FEATURES
# =========================================================
pca_feature_bases = [b for b in feature_bases if not is_brainsegvol_feature(b)]
print("PCA feature bases after excluding BrainSegVol-like features:", len(pca_feature_bases))

# Near-zero variance filtering based on baseline data only.
valid_feature_names = []
for b in pca_feature_bases:
    pre_col = f"{b}_pre"
    s = pd.to_numeric(df[pre_col], errors="coerce")
    if s.notna().sum() >= 3:
        v = np.nanvar(s.to_numpy(dtype=float))
        if np.isfinite(v) and v > variance_threshold:
            valid_feature_names.append(b)

pca_feature_bases = [b for b in pca_feature_bases if b in valid_feature_names]
print("PCA feature bases after baseline variance filter:", len(pca_feature_bases))

if len(pca_feature_bases) < 2:
    raise ValueError("Not enough variable whole-brain features for PCA after filtering.")


# =========================================================
# BUILD LONG DATA
# =========================================================
meta_cols = [subject_col]
for col in ["cohort", "id", "sex", "age", dose_col] + poly_file_cols:
    if col in df.columns and col not in meta_cols:
        meta_cols.append(col)

pre_long = df[meta_cols + [brainsegvol_pre_col] + [f"{b}_pre" for b in pca_feature_bases]].copy()
pre_long["time"] = 0
pre_long["session"] = "sessie1"
pre_long = pre_long.rename(columns={brainsegvol_pre_col: "BrainSegVol"})
pre_long = pre_long.rename(columns={f"{b}_pre": b for b in pca_feature_bases})

post_long = df[meta_cols + [brainsegvol_post_col] + [f"{b}_post" for b in pca_feature_bases]].copy()
post_long["time"] = 1
post_long["session"] = "sessie3"
post_long = post_long.rename(columns={brainsegvol_post_col: "BrainSegVol"})
post_long = post_long.rename(columns={f"{b}_post": b for b in pca_feature_bases})

long_df = pd.concat([pre_long, post_long], axis=0, ignore_index=True)


# =========================================================
# IMPUTE AGE
# =========================================================
if "age" in long_df.columns:
    long_df["age"] = pd.to_numeric(long_df["age"], errors="coerce")
    mean_age = long_df["age"].mean(skipna=True)
    long_df["age"] = long_df["age"].fillna(mean_age)
    print(f"Imputed missing age with mean age = {mean_age:.3f}")


# =========================================================
# DROP PCA FEATURES WITH ANY MISSING VALUES ACROSS BASELINE/FOLLOW-UP
# This keeps the same feature set for baseline fitting and follow-up projection.
# =========================================================
missing_prop = long_df[pca_feature_bases].isna().mean()
bad_features = missing_prop[missing_prop > 0].index.tolist()

print(f"Dropping {len(bad_features)} PCA features with > 0.0% missingness")
print(bad_features)

pca_feature_bases = [f for f in pca_feature_bases if f not in bad_features]
print("Remaining PCA features:", len(pca_feature_bases))

if len(pca_feature_bases) < 2:
    raise ValueError("Not enough PCA features after missingness filtering.")


# =========================================================
# COMPLETE-CASE FILTER FOR PCA + PRIMARY MODEL
# Polysubstance complete cases are handled separately for sensitivity model.
# =========================================================
needed_cols_for_pca = [
    subject_col, "time", "session", "BrainSegVol", "age", "sex", dose_col
] + pca_feature_bases
needed_cols_for_pca = [c for c in needed_cols_for_pca if c in long_df.columns]

long_complete = long_df.dropna(subset=needed_cols_for_pca).copy()

print("Long complete data before paired-subject filter:", long_complete.shape)
print("Unique subjects before paired-subject filter:", long_complete[subject_col].nunique())

pair_counts = long_complete.groupby(subject_col)["time"].nunique()
paired_subjects = pair_counts[pair_counts == 2].index.tolist()
print("Paired subjects retained:", len(paired_subjects))

long_complete = long_complete[long_complete[subject_col].isin(paired_subjects)].copy()
long_complete = long_complete.sort_values([subject_col, "time"]).reset_index(drop=True)

if len(paired_subjects) < 3:
    raise ValueError("Too few paired subjects for mixed model.")


# =========================================================
# BASELINE-ONLY PCA + FOLLOW-UP PROJECTION
# =========================================================
base_df = long_complete[long_complete["time"] == 0].sort_values(subject_col).copy()
follow_df = long_complete[long_complete["time"] == 1].sort_values(subject_col).copy()

base_subjects = base_df[subject_col].astype(str).tolist()
follow_subjects = follow_df[subject_col].astype(str).tolist()
if base_subjects != follow_subjects:
    raise ValueError("Baseline and follow-up subject order mismatch after sorting.")

X_base = base_df[pca_feature_bases].to_numpy(dtype=float)
X_follow = follow_df[pca_feature_bases].to_numpy(dtype=float)

scaler = StandardScaler()
X_base_scaled = scaler.fit_transform(X_base)
X_follow_scaled = scaler.transform(X_follow)

n_components = min(n_pcs_to_save, X_base_scaled.shape[1], X_base_scaled.shape[0])
pca = PCA(n_components=n_components, random_state=42)
base_scores = pca.fit_transform(X_base_scaled)
follow_scores = pca.transform(X_follow_scaled)

pc_cols = [f"PC{i+1}" for i in range(n_components)]

scores_df = pd.DataFrame(index=long_complete.index, columns=pc_cols, dtype=float)
scores_df.loc[base_df.index, pc_cols] = base_scores
scores_df.loc[follow_df.index, pc_cols] = follow_scores

long_complete = pd.concat([long_complete, scores_df], axis=1)

# Save PCA outputs.
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

# Save baseline means/scales for reproducibility.
scaler_df = pd.DataFrame({
    "feature": pca_feature_bases,
    "baseline_scaler_mean": scaler.mean_,
    "baseline_scaler_scale": scaler.scale_,
})

long_pca_path = os.path.join(output_root, "whole_brain_long_with_baselinePCA_scores_and_dose.csv")
loadings_path = os.path.join(output_root, "whole_brain_baselinePCA_loadings.csv")
explained_path = os.path.join(output_root, "whole_brain_baselinePCA_explained_variance.csv")
scaler_path = os.path.join(output_root, "whole_brain_baselinePCA_scaler_mean_scale.csv")

long_complete.to_csv(long_pca_path, index=False)
loadings.to_csv(loadings_path, index=False)
explained_var_df.to_csv(explained_path, index=False)
scaler_df.to_csv(scaler_path, index=False)

print(f"Saved: {long_pca_path}")
print(f"Saved: {loadings_path}")
print(f"Saved: {explained_path}")
print(f"Saved: {scaler_path}")


# =========================================================
# PREPARE DOSE + COVARIATES FOR MODEL
# =========================================================
long_complete[dose_col] = pd.to_numeric(long_complete[dose_col], errors="coerce")

if use_log_dose:
    long_complete["dose_xtc"] = np.log1p(long_complete[dose_col])
    print(f"Using log-transformed dose: log1p({dose_col})")
else:
    long_complete["dose_xtc"] = long_complete[dose_col]
    print(f"Using raw dose: {dose_col}")

# Center primary continuous covariates.
long_complete["age_c"] = long_complete["age"] - long_complete["age"].mean()
long_complete["BrainSegVol_c"] = long_complete["BrainSegVol"] - long_complete["BrainSegVol"].mean()
long_complete["dose_xtc_c"] = long_complete["dose_xtc"] - long_complete["dose_xtc"].mean()
long_complete["sex"] = long_complete["sex"].astype("category")

# Rename polysubstance columns to readable names and convert to numeric.
for clean_name, raw_col in polysub_cols.items():
    if raw_col in long_complete.columns:
        long_complete[clean_name] = pd.to_numeric(long_complete[raw_col], errors="coerce")

# Center polysubstance covariates.
available_polysub = []
for clean_name in polysub_cols.keys():
    if clean_name in long_complete.columns:
        centered_name = f"{clean_name}_c"
        long_complete[centered_name] = long_complete[clean_name] - long_complete[clean_name].mean()
        available_polysub.append(centered_name)

print("Available polysubstance covariates for sensitivity model:")
print(available_polysub)


# =========================================================
# OPTIONAL POLYDRUG-USER FLAG
# =========================================================
long_complete["non_xtc_polydrug_user"] = False
available_for_exclusion = []

for clean_name in polydrug_exclusion_clean_names:
    if clean_name in long_complete.columns:
        available_for_exclusion.append(clean_name)
        no_use_mask = is_nonuse_or_missing_like(long_complete[clean_name])
        long_complete["non_xtc_polydrug_user"] = (
            long_complete["non_xtc_polydrug_user"] | (~no_use_mask.fillna(False))
        )

print("Polydrug exclusion variables:", available_for_exclusion)
print(
    "Subjects flagged as non-XTC polydrug users:",
    long_complete.loc[long_complete["non_xtc_polydrug_user"], subject_col].nunique()
)


# =========================================================
# PRIMARY IMAGING MODEL
# PC1 ~ time * XTC_dose + age + sex + BrainSegVol + (1 | subject)
# =========================================================
primary_cols = [
    subject_col, "PC1", "time", "dose_xtc_c",
    "age_c", "sex", "BrainSegVol_c"
]

lmm_primary_df = long_complete[primary_cols].dropna().copy()

print("\nPrimary model df:", lmm_primary_df.shape)
print("Primary model unique subjects:", lmm_primary_df[subject_col].nunique())

primary_formula = "PC1 ~ time * dose_xtc_c + age_c + C(sex) + BrainSegVol_c"
print("Primary formula:", primary_formula)

lmm_primary_res = fit_mixedlm_with_fallback(
    primary_formula,
    data=lmm_primary_df,
    group_col=subject_col
)

primary_params = save_lmm_outputs(
    lmm_primary_res,
    output_root,
    prefix="whole_brain_pc1_baselinePCA_lmm_primary_xtc_dose"
)

# Backward-compatible output names if needed.
with open(os.path.join(output_root, "whole_brain_pc1_lmm_with_dose_summary.txt"), "w") as f:
    f.write(lmm_primary_res.summary().as_text())
primary_params.to_csv(
    os.path.join(output_root, "whole_brain_pc1_lmm_with_dose_coefficients.csv"),
    index=False
)


# =========================================================
# SENSITIVITY MODEL 1: POLYSUBSTANCE-ADJUSTED
# =========================================================
if len(available_polysub) > 0:
    sensitivity_cols = primary_cols + available_polysub
    lmm_sens1_df = long_complete[sensitivity_cols].dropna().copy()

    print("\nSensitivity model 1, polysubstance-adjusted df:", lmm_sens1_df.shape)
    print("Sensitivity model 1 unique subjects:", lmm_sens1_df[subject_col].nunique())

    if lmm_sens1_df[subject_col].nunique() < 3:
        print("Too few subjects for sensitivity model 1 after polysubstance complete-case filtering.")
        lmm_sens1_res = None
        sens1_params = None
    else:
        polysub_formula_terms = " + ".join(available_polysub)
        sensitivity_formula = (
            "PC1 ~ time * dose_xtc_c "
            "+ age_c + C(sex) + BrainSegVol_c "
            f"+ {polysub_formula_terms}"
        )
        print("Sensitivity model 1 formula:", sensitivity_formula)

        lmm_sens1_res = fit_mixedlm_with_fallback(
            sensitivity_formula,
            data=lmm_sens1_df,
            group_col=subject_col
        )

        sens1_params = save_lmm_outputs(
            lmm_sens1_res,
            output_root,
            prefix="whole_brain_pc1_baselinePCA_lmm_sensitivity1_polysub_adjusted"
        )
else:
    print("No polysubstance columns available. Sensitivity model 1 was not fitted.")
    lmm_sens1_res = None
    sens1_params = None


# =========================================================
# SENSITIVITY MODEL 2: EXCLUDE NON-XTC POLYDRUG USERS
# This is optional and depends strongly on the chosen definition above.
# =========================================================
if run_polydrug_exclusion_sensitivity and len(available_for_exclusion) > 0:
    sens2_cols = primary_cols + ["non_xtc_polydrug_user"]
    lmm_sens2_df = long_complete[sens2_cols].dropna().copy()
    lmm_sens2_df = lmm_sens2_df[lmm_sens2_df["non_xtc_polydrug_user"] == False].copy()

    print("\nSensitivity model 2, excluding non-XTC polydrug users df:", lmm_sens2_df.shape)
    print("Sensitivity model 2 unique subjects:", lmm_sens2_df[subject_col].nunique())
    print("Sensitivity model 2 excludes users of:", available_for_exclusion)

    if lmm_sens2_df[subject_col].nunique() < 3:
        print("Too few subjects for sensitivity model 2 after polydrug-user exclusion.")
        lmm_sens2_res = None
        sens2_params = None
    elif lmm_sens2_df["dose_xtc_c"].nunique(dropna=True) < 2:
        print("Not enough XTC-dose variation for sensitivity model 2 after polydrug-user exclusion.")
        lmm_sens2_res = None
        sens2_params = None
    else:
        sens2_formula = primary_formula
        print("Sensitivity model 2 formula:", sens2_formula)

        lmm_sens2_res = fit_mixedlm_with_fallback(
            sens2_formula,
            data=lmm_sens2_df,
            group_col=subject_col
        )

        sens2_params = save_lmm_outputs(
            lmm_sens2_res,
            output_root,
            prefix="whole_brain_pc1_baselinePCA_lmm_sensitivity2_exclude_polydrug_users"
        )
else:
    print("Sensitivity model 2 was not fitted.")
    lmm_sens2_res = None
    sens2_params = None


# =========================================================
# SUBJECT-LEVEL PC1 DELTA FILE
# =========================================================
wide_index_cols = [
    c for c in [subject_col, "cohort", "id", "age", "sex", dose_col]
    if c in long_complete.columns
]

for c in wide_index_cols:
    if str(long_complete[c].dtype) == "category":
        long_complete[c] = long_complete[c].astype(str)

pc_wide = (
    long_complete[wide_index_cols + ["session"] + pc_cols]
    .drop_duplicates(subset=wide_index_cols + ["session"])
    .pivot(
        index=wide_index_cols,
        columns="session",
        values=pc_cols
    )
)

# Flatten MultiIndex columns: PC1_sessie1, PC1_sessie3, etc.
pc_wide.columns = [f"{pc}_{sess}" for pc, sess in pc_wide.columns]
pc_wide = pc_wide.reset_index()

for pc in pc_cols:
    pre_col = f"{pc}_sessie1"
    post_col = f"{pc}_sessie3"
    delta_col = f"{pc}_delta"
    if {pre_col, post_col}.issubset(pc_wide.columns):
        pc_wide[delta_col] = pc_wide[post_col] - pc_wide[pre_col]
    else:
        pc_wide[delta_col] = np.nan
        print(f"WARNING: {pre_col} and/or {post_col} not found in pc_wide columns.")

pc_wide_out = os.path.join(
    output_root,
    "whole_brain_baselinePCA_subject_wide_with_dose.csv"
)
pc_wide.to_csv(pc_wide_out, index=False)
print("Saved:", pc_wide_out)


# =========================================================
# PRINT SUMMARY
# =========================================================
print("\nFinished whole-brain baseline-PCA + LMM with XTC dose.")
print("Saved to:", output_root)

print("\nPCA explained variance from baseline-fitted PCA:")
print(explained_var_df.to_string(index=False))

print("\nPrimary LMM coefficients:")
print(primary_params.to_string(index=False))

if sens1_params is not None:
    print("\nSensitivity model 1, polysubstance-adjusted coefficients:")
    print(sens1_params.to_string(index=False))

if sens2_params is not None:
    print("\nSensitivity model 2, excluding non-XTC polydrug users coefficients:")
    print(sens2_params.to_string(index=False))

print("\nKey interpretation guide:")
print("- PCA is fitted on baseline only; follow-up is projected into the baseline-defined PCA space.")
print("- time                 = overall Session 1 -> Session 3 shift in PC1")
print("- dose_xtc_c           = baseline difference by XTC dose")
print("- time:dose_xtc_c      = whether higher XTC dose is associated with larger longitudinal PC1 change")
print("- sensitivity model 1  = same key term, additionally adjusted for polysubstance variables")
print("- sensitivity model 2  = same key term after excluding selected non-XTC polydrug users")
