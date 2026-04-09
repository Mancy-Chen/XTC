# Mingshi Chen
# Whole-brain PCA + Linear Mixed Model with XTC dosage
#
# Model:
# PC1 ~ time + dose_xtc + time:dose_xtc + age + sex + BrainSegVol + (1 | subject_id)
#
# Inputs:
# 1) whole_brain_volume_pre_delta.csv
# 2) xtc_dose_wide_mapped_to_y.csv
#
# Notes:
# - Session 3 is reconstructed as pre + delta
# - Age is mean-imputed
# - PCA excludes BrainSegVol features
# - PCA features with any missing values are dropped
# - Choose your dosage variable via dose_col below

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
dose_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/xtc_dose_wide_mapped_to_y.csv"

output_root = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_lmm_with_xtc_dose"
os.makedirs(output_root, exist_ok=True)

subject_col = "subject_id"

# ---------------------------------------------------------
# Choose the dosage variable you want to test
# Change this depending on your file columns
# Examples:
#   "dose_post"
#   "dose_sum_1_2_3"
#   "dose_change_1_to_3"
#   "xlttot_sessie3"
# ---------------------------------------------------------
dose_col = "xlttot_sessie3"

# log-transform dose if dose is skewed
use_log_dose = True

# brain volume covariate pattern
brainsegvol_patterns = [
    "brainsegvol",
    "brainsegvolnotvent",
]

variance_threshold = 1e-12

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
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object:
            df[c] = safe_numeric(df[c])
    return df

def pick_feature_bases(df):
    all_cols = df.columns.tolist()
    pre_cols = [c for c in all_cols if c.endswith("_pre")]
    delta_cols = [c for c in all_cols if c.endswith("_delta")]

    pre_bases = {c[:-4] for c in pre_cols}
    delta_bases = {c[:-6] for c in delta_cols}
    common_bases = sorted(pre_bases & delta_bases)

    exclude_exact = {
        "cohort", "id", "age", "sex", "session", "subject_id", "subject_code"
    }
    common_bases = [b for b in common_bases if b not in exclude_exact]
    return common_bases

def is_brainsegvol_feature(base):
    s = base.lower()
    return any(p in s for p in brainsegvol_patterns)

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
# KEEP DOSAGE COLUMNS
# =========================================================
dose_keep_cols = [c for c in ["subject_code", "cohort_x", "vwrec", dose_col] if c in dose_df.columns]
dose_small = dose_df[dose_keep_cols].copy()

dose_small["subject_code"] = dose_small["subject_code"].astype(str).str.strip()
X_df["subject_id"] = X_df["subject_id"].astype(str).str.strip()

print("Merging X subject_id with dose subject_code")

df = X_df.merge(
    dose_small,
    left_on="subject_id",
    right_on="subject_code",
    how="left"
)

# guarantee subject_id exists after merge
if subject_col not in df.columns and {"cohort", "id"}.issubset(df.columns):
    cohort_map = {1: "I", 2: "II", 3: "III", 4: "IV"}
    df[subject_col] = df.apply(
        lambda r: f"{cohort_map.get(int(r['cohort']), str(int(r['cohort'])))}{int(r['id']):03d}"
        if pd.notna(r["cohort"]) and pd.notna(r["id"]) else np.nan,
        axis=1
    )

print("Merged df:", df.shape)

# =========================================================
# RECONSTRUCT POST
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
# FIND BrainSegVol
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

pre_feature_cols = [f"{b}_pre" for b in pca_feature_bases]
post_feature_cols = [f"{b}_post" for b in pca_feature_bases]

# remove near-zero variance features
stack_for_var = pd.concat(
    [
        df[pre_feature_cols].rename(columns=lambda c: c[:-4]),
        df[post_feature_cols].rename(columns=lambda c: c[:-5]),
    ],
    axis=0,
    ignore_index=True
)

valid_feature_names = []
for col in stack_for_var.columns:
    s = pd.to_numeric(stack_for_var[col], errors="coerce")
    if s.notna().sum() >= 3:
        v = np.nanvar(s.to_numpy(dtype=float))
        if np.isfinite(v) and v > variance_threshold:
            valid_feature_names.append(col)

pca_feature_bases = [b for b in pca_feature_bases if b in valid_feature_names]
print("PCA feature bases after variance filter:", len(pca_feature_bases))

# =========================================================
# BUILD LONG DATA
# =========================================================
meta_cols = [subject_col]
for col in ["cohort", "id", "sex", "age", dose_col]:
    if col in df.columns:
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
# DROP PCA FEATURES WITH ANY MISSING VALUES
# =========================================================
missing_prop = long_df[pca_feature_bases].isna().mean()
bad_features = missing_prop[missing_prop > 0].index.tolist()

print(f"Dropping {len(bad_features)} PCA features with > 0.0% missingness")
print(bad_features)

pca_feature_bases = [f for f in pca_feature_bases if f not in bad_features]
print("Remaining PCA features:", len(pca_feature_bases))

# =========================================================
# COMPLETE CASE FILTER
# =========================================================
needed_cols = [subject_col, "time", "session", "BrainSegVol", "age", "sex", dose_col] + pca_feature_bases
needed_cols = [c for c in needed_cols if c in long_df.columns]

long_complete = long_df.dropna(subset=needed_cols).copy()

print("Long complete data:", long_complete.shape)
print("Unique subjects for PCA/LMM:", long_complete[subject_col].nunique())

pair_counts = long_complete.groupby(subject_col)["time"].nunique()
paired_subjects = pair_counts[pair_counts == 2].index.tolist()
print("Paired subjects retained:", len(paired_subjects))

long_complete = long_complete[long_complete[subject_col].isin(paired_subjects)].copy()

if len(paired_subjects) < 3:
    raise ValueError("Too few paired subjects for mixed model.")

# =========================================================
# PCA
# =========================================================
X_pca = long_complete[pca_feature_bases].to_numpy(dtype=float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=min(5, X_scaled.shape[1], X_scaled.shape[0]), random_state=42)
scores = pca.fit_transform(X_scaled)

pc_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
scores_df = pd.DataFrame(scores, columns=pc_cols, index=long_complete.index)

long_complete = pd.concat(
    [long_complete.reset_index(drop=True), scores_df.reset_index(drop=True)],
    axis=1
)

# save PCA outputs
loadings = pd.DataFrame(
    pca.components_.T,
    index=pca_feature_bases,
    columns=pc_cols
).reset_index().rename(columns={"index": "feature"})

explained_var_df = pd.DataFrame({
    "component": pc_cols,
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_)
})

long_complete.to_csv(os.path.join(output_root, "whole_brain_long_with_pca_scores_and_dose.csv"), index=False)
loadings.to_csv(os.path.join(output_root, "whole_brain_pca_loadings.csv"), index=False)
explained_var_df.to_csv(os.path.join(output_root, "whole_brain_pca_explained_variance.csv"), index=False)

# =========================================================
# PREPARE DOSE FOR MODEL
# =========================================================
long_complete[dose_col] = pd.to_numeric(long_complete[dose_col], errors="coerce")

if use_log_dose:
    long_complete["dose_xtc"] = np.log1p(long_complete[dose_col])
    print(f"Using log-transformed dose: log1p({dose_col})")
else:
    long_complete["dose_xtc"] = long_complete[dose_col]
    print(f"Using raw dose: {dose_col}")

# center continuous covariates
long_complete["age_c"] = long_complete["age"] - long_complete["age"].mean()
long_complete["BrainSegVol_c"] = long_complete["BrainSegVol"] - long_complete["BrainSegVol"].mean()
long_complete["dose_xtc_c"] = long_complete["dose_xtc"] - long_complete["dose_xtc"].mean()

long_complete["sex"] = long_complete["sex"].astype("category")

# =========================================================
# LMM: PC1 ~ time * dose_xtc + age + sex + BrainSegVol + (1|subject)
# =========================================================
lmm_df = long_complete[
    [subject_col, "PC1", "time", "dose_xtc_c", "age_c", "sex", "BrainSegVol_c"]
].dropna().copy()

model_formula = "PC1 ~ time * dose_xtc_c + age_c + C(sex) + BrainSegVol_c"

try:
    lmm = smf.mixedlm(
        model_formula,
        data=lmm_df,
        groups=lmm_df[subject_col],
        re_formula="1"
    )
    lmm_res = lmm.fit(reml=False, method="lbfgs")
except Exception:
    lmm = smf.mixedlm(
        model_formula,
        data=lmm_df,
        groups=lmm_df[subject_col]
    )
    lmm_res = lmm.fit(reml=False)

# =========================================================
# SAVE MODEL OUTPUT
# =========================================================
with open(os.path.join(output_root, "whole_brain_pc1_lmm_with_dose_summary.txt"), "w") as f:
    f.write(lmm_res.summary().as_text())

lmm_params = pd.DataFrame({
    "term": lmm_res.params.index,
    "beta": lmm_res.params.values,
    "p_value": lmm_res.pvalues.reindex(lmm_res.params.index).values,
    "std_error": lmm_res.bse.reindex(lmm_res.params.index).values,
    "z_value": lmm_res.tvalues.reindex(lmm_res.params.index).values,
})
lmm_params.to_csv(
    os.path.join(output_root, "whole_brain_pc1_lmm_with_dose_coefficients.csv"),
    index=False
)

# =========================================================
# OPTIONAL: SUBJECT-LEVEL PC1 DELTA FILE
# =========================================================
wide_index_cols = [c for c in [subject_col, "cohort", "id", "age", "sex", dose_col] if c in long_complete.columns]
pc1_wide = long_complete.pivot_table(
    index=wide_index_cols,
    columns="session",
    values="PC1"
).reset_index()

if {"sessie1", "sessie3"}.issubset(pc1_wide.columns):
    pc1_wide["PC1_delta"] = pc1_wide["sessie3"] - pc1_wide["sessie1"]

pc1_wide.to_csv(os.path.join(output_root, "whole_brain_pc1_subject_wide_with_dose.csv"), index=False)

# =========================================================
# PRINT SUMMARY
# =========================================================
print("\nFinished whole-brain PCA + LMM with XTC dose.")
print("Saved to:", output_root)

print("\nPCA explained variance:")
print(explained_var_df.to_string(index=False))

print("\nLMM coefficients:")
print(lmm_params.to_string(index=False))

print("\nKey interpretation guide:")
print("- time                 = overall Session 1 -> Session 3 shift in PC1")
print("- dose_xtc_c           = baseline difference by dose")
print("- time:dose_xtc_c      = whether higher dose is associated with larger longitudinal change")

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
vwrec_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/xtc_dose_wide_mapped_to_y.csv"

output_root = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_lmm_with_vwrec"
os.makedirs(output_root, exist_ok=True)

subject_col = "subject_id"
predictor_col = "vwrec"

brainsegvol_patterns = [
    "brainsegvol",
    "brainsegvolnotvent",
]

variance_threshold = 1e-12

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
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object:
            df[c] = safe_numeric(df[c])
    return df

def pick_feature_bases(df):
    all_cols = df.columns.tolist()
    pre_cols = [c for c in all_cols if c.endswith("_pre")]
    delta_cols = [c for c in all_cols if c.endswith("_delta")]

    pre_bases = {c[:-4] for c in pre_cols}
    delta_bases = {c[:-6] for c in delta_cols}
    common_bases = sorted(pre_bases & delta_bases)

    exclude_exact = {
        "cohort", "id", "age", "sex", "session", "subject_id", "subject_code", "vwrec"
    }
    common_bases = [b for b in common_bases if b not in exclude_exact]
    return common_bases

def is_brainsegvol_feature(base):
    s = base.lower()
    return any(p in s for p in brainsegvol_patterns)

# =========================================================
# LOAD
# =========================================================
X_df = pd.read_csv(x_path)
pred_df = pd.read_csv(vwrec_path)

X_df.columns = [c.strip() for c in X_df.columns]
pred_df.columns = [c.strip() for c in pred_df.columns]

X_df = add_subject_id_from_cohort_id(X_df)
X_df = add_cohort_id_from_subject_id(X_df)

pred_df = add_subject_id_from_cohort_id(pred_df)
pred_df = add_cohort_id_from_subject_id(pred_df)

X_df = prepare_numeric_columns(X_df, exclude={subject_col})
pred_df = prepare_numeric_columns(pred_df, exclude={subject_col, "subject_code"})

print("Loaded X:", X_df.shape)
print("Loaded predictor file:", pred_df.shape)
print("Predictor file columns:", pred_df.columns.tolist())

if predictor_col not in pred_df.columns:
    raise ValueError(f"{predictor_col} not found in predictor file.")

# =========================================================
# MERGE USING subject_id <-> subject_code
# =========================================================
pred_small = pred_df[[c for c in ["subject_code", predictor_col] if c in pred_df.columns]].copy()

pred_small["subject_code"] = pred_small["subject_code"].astype(str).str.strip()
X_df["subject_id"] = X_df["subject_id"].astype(str).str.strip()

df = X_df.merge(
    pred_small,
    left_on="subject_id",
    right_on="subject_code",
    how="left"
)

print("Merged df:", df.shape)
print(f"Missing {predictor_col} rows:", df[predictor_col].isna().sum())

# =========================================================
# RECONSTRUCT POST
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
# FIND BrainSegVol
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
# PREP PCA FEATURES
# =========================================================
pca_feature_bases = [b for b in feature_bases if not is_brainsegvol_feature(b)]
print("PCA feature bases after excluding BrainSegVol-like features:", len(pca_feature_bases))

pre_feature_cols = [f"{b}_pre" for b in pca_feature_bases]
post_feature_cols = [f"{b}_post" for b in pca_feature_bases]

stack_for_var = pd.concat(
    [
        df[pre_feature_cols].rename(columns=lambda c: c[:-4]),
        df[post_feature_cols].rename(columns=lambda c: c[:-5]),
    ],
    axis=0,
    ignore_index=True
)

valid_feature_names = []
for col in stack_for_var.columns:
    s = pd.to_numeric(stack_for_var[col], errors="coerce")
    if s.notna().sum() >= 3:
        v = np.nanvar(s.to_numpy(dtype=float))
        if np.isfinite(v) and v > variance_threshold:
            valid_feature_names.append(col)

pca_feature_bases = [b for b in pca_feature_bases if b in valid_feature_names]
print("PCA feature bases after variance filter:", len(pca_feature_bases))

# =========================================================
# BUILD LONG DATA
# =========================================================
meta_cols = [subject_col]
for col in ["cohort", "id", "sex", "age", predictor_col]:
    if col in df.columns:
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

# mean-impute age
if "age" in long_df.columns:
    long_df["age"] = pd.to_numeric(long_df["age"], errors="coerce")
    mean_age = long_df["age"].mean(skipna=True)
    long_df["age"] = long_df["age"].fillna(mean_age)
    print(f"Imputed missing age with mean age = {mean_age:.3f}")

# drop PCA features with any missingness
missing_prop = long_df[pca_feature_bases].isna().mean()
bad_features = missing_prop[missing_prop > 0].index.tolist()

print(f"Dropping {len(bad_features)} PCA features with >0 missingness")
print(bad_features)

pca_feature_bases = [f for f in pca_feature_bases if f not in bad_features]
print("Remaining PCA features:", len(pca_feature_bases))

# complete-case filter
needed_cols = [subject_col, "time", "session", "BrainSegVol", "age", "sex", predictor_col] + pca_feature_bases
long_complete = long_df.dropna(subset=needed_cols).copy()

print("Long complete data:", long_complete.shape)
print("Unique subjects for PCA/LMM:", long_complete[subject_col].nunique())

pair_counts = long_complete.groupby(subject_col)["time"].nunique()
paired_subjects = pair_counts[pair_counts == 2].index.tolist()
print("Paired subjects retained:", len(paired_subjects))

long_complete = long_complete[long_complete[subject_col].isin(paired_subjects)].copy()

# =========================================================
# PCA
# =========================================================
X_pca = long_complete[pca_feature_bases].to_numpy(dtype=float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=min(5, X_scaled.shape[1], X_scaled.shape[0]), random_state=42)
scores = pca.fit_transform(X_scaled)

pc_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
scores_df = pd.DataFrame(scores, columns=pc_cols, index=long_complete.index)

long_complete = pd.concat(
    [long_complete.reset_index(drop=True), scores_df.reset_index(drop=True)],
    axis=1
)

# =========================================================
# LMM WITH vwrec
# =========================================================
long_complete[predictor_col] = pd.to_numeric(long_complete[predictor_col], errors="coerce")
long_complete["vwrec_c"] = long_complete[predictor_col] - long_complete[predictor_col].mean()
long_complete["age_c"] = long_complete["age"] - long_complete["age"].mean()
long_complete["BrainSegVol_c"] = long_complete["BrainSegVol"] - long_complete["BrainSegVol"].mean()
long_complete["sex"] = long_complete["sex"].astype("category")

lmm_df = long_complete[
    [subject_col, "PC1", "time", "vwrec_c", "age_c", "sex", "BrainSegVol_c"]
].dropna().copy()

model_formula = "PC1 ~ time * vwrec_c + age_c + C(sex) + BrainSegVol_c"

lmm = smf.mixedlm(
    model_formula,
    data=lmm_df,
    groups=lmm_df[subject_col],
    re_formula="1"
)

res = lmm.fit(reml=False, method="lbfgs")

with open(os.path.join(output_root, "whole_brain_pc1_lmm_with_vwrec_summary.txt"), "w") as f:
    f.write(res.summary().as_text())

coef_df = pd.DataFrame({
    "term": res.params.index,
    "beta": res.params.values,
    "p_value": res.pvalues.reindex(res.params.index).values,
    "std_error": res.bse.reindex(res.params.index).values,
    "z_value": res.tvalues.reindex(res.params.index).values,
})
coef_df.to_csv(os.path.join(output_root, "whole_brain_pc1_lmm_with_vwrec_coefficients.csv"), index=False)

print("\nFinished LMM with vwrec.")
print(coef_df)