# Mingshi Chen 25-03-2026
# Whole-brain volume PCA + linear mixed-effects model
#
# Goal:
# 1) Reconstruct post = pre + delta from whole_brain_volume_pre_delta.csv
# 2) Build whole-brain PCA scores from volumetric features
# 3) Test sessie1 vs sessie3 change using a linear mixed-effects model
#    with covariates: age, sex, BrainSegVol
# 4) Test whether PCA change is associated with vwrec
#
# Notes:
# - Session 3 is reconstructed from pre + delta
# - Missing age is mean-imputed
# - Diagnostic output is included to show where subjects are lost

import os
import re
import warnings
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# =========================================================
# SETTINGS
# =========================================================
x_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/whole_brain_volume_pre_delta.csv"
y_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/y_with_id_filteredQC.csv"

output_root = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_volume_pca_lmm_mean_age_imputed"
os.makedirs(output_root, exist_ok=True)

outcome_col = "vwrec"
subject_col = "subject_id"

# BrainSegVol is used as a covariate, so exclude it from PCA features
brainsegvol_patterns = [
    "brainsegvol",
    "brainsegvolnotvent",
]

# Optional: features with near-zero variance can destabilize PCA
variance_threshold = 1e-12

# =========================================================
# HELPERS
# =========================================================
def safe_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )

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

def infer_merge_keys(X_df, y_df):
    if {"cohort", "id"}.issubset(X_df.columns) and {"cohort", "id"}.issubset(y_df.columns):
        return ["cohort", "id"]
    if subject_col in X_df.columns and subject_col in y_df.columns:
        return [subject_col]
    return None

def pick_feature_bases(df):
    all_cols = df.columns.tolist()
    pre_cols = [c for c in all_cols if c.endswith("_pre")]
    delta_cols = [c for c in all_cols if c.endswith("_delta")]

    pre_bases = {c[:-4] for c in pre_cols}
    delta_bases = {c[:-6] for c in delta_cols}
    common_bases = sorted(pre_bases & delta_bases)

    exclude_exact = {
        "cohort", "id", "age", "sex", "vwrec", "session", "subject_id", "subject_code"
    }
    common_bases = [b for b in common_bases if b not in exclude_exact]
    return common_bases

def is_brainsegvol_feature(base):
    s = base.lower()
    return any(p in s for p in brainsegvol_patterns)

def prepare_numeric_columns(df, exclude=None):
    exclude = set() if exclude is None else set(exclude)
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object:
            df[c] = safe_numeric(df[c])
    return df

# =========================================================
# LOAD
# =========================================================
X_df = pd.read_csv(x_path)
y_df = pd.read_csv(y_path)

X_df.columns = [c.strip() for c in X_df.columns]
y_df.columns = [c.strip() for c in y_df.columns]

X_df = add_subject_id_from_cohort_id(X_df)
X_df = add_cohort_id_from_subject_id(X_df)

y_df = add_subject_id_from_cohort_id(y_df)
y_df = add_cohort_id_from_subject_id(y_df)

y_df[outcome_col] = safe_numeric(y_df[outcome_col])

X_df = prepare_numeric_columns(X_df, exclude={subject_col})
y_df = prepare_numeric_columns(y_df, exclude={subject_col})

print("Loaded X:", X_df.shape)
print("Loaded y:", y_df.shape)

merge_keys = infer_merge_keys(X_df, y_df)
if merge_keys is None:
    raise ValueError(
        "Could not infer merge keys. Need either cohort+id or subject_id in both X and y."
    )

print("Merging on:", merge_keys)
y_keep = list(dict.fromkeys(merge_keys + [outcome_col]))
df = X_df.merge(y_df[y_keep], on=merge_keys, how="left")

# subject_id should always be present after merge
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

# Find BrainSegVol covariate columns
brainsegvol_base_candidates = [b for b in feature_bases if is_brainsegvol_feature(b)]
if len(brainsegvol_base_candidates) == 0:
    raise ValueError(
        "No BrainSegVol-like feature found. Expected something like aseg+DKT_BrainSegVol_pre/delta."
    )

# Prefer exact BrainSegVol before BrainSegVolNotVent if both exist
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
# BUILD LONG DATA FOR PCA + LMM
# =========================================================
pca_feature_bases = [b for b in feature_bases if not is_brainsegvol_feature(b)]
print("PCA feature bases after excluding BrainSegVol-like features:", len(pca_feature_bases))

pre_feature_cols = [f"{b}_pre" for b in pca_feature_bases]
post_feature_cols = [f"{b}_post" for b in pca_feature_bases]

# remove columns with near-zero variance across stacked pre/post data
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
pre_feature_cols = [f"{b}_pre" for b in pca_feature_bases]
post_feature_cols = [f"{b}_post" for b in pca_feature_bases]

print("PCA feature bases after variance filter:", len(pca_feature_bases))

if len(pca_feature_bases) < 2:
    raise ValueError("Not enough variable whole-brain features for PCA after filtering.")

meta_cols = [subject_col]
for col in ["cohort", "id", "sex", "age", outcome_col]:
    if col in df.columns:
        meta_cols.append(col)

pre_long = df[meta_cols + [brainsegvol_pre_col] + pre_feature_cols].copy()
pre_long["time"] = 0
pre_long["session"] = "sessie1"
pre_long = pre_long.rename(columns={brainsegvol_pre_col: "BrainSegVol"})
pre_long = pre_long.rename(columns={f"{b}_pre": b for b in pca_feature_bases})

post_long = df[meta_cols + [brainsegvol_post_col] + post_feature_cols].copy()
post_long["time"] = 1
post_long["session"] = "sessie3"
post_long = post_long.rename(columns={brainsegvol_post_col: "BrainSegVol"})
post_long = post_long.rename(columns={f"{b}_post": b for b in pca_feature_bases})

long_df = pd.concat([pre_long, post_long], axis=0, ignore_index=True)

# =========================================================
# MEAN IMPUTATION FOR AGE
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
# DIAGNOSTICS FOR MISSINGNESS
# =========================================================
print("\n===== DIAGNOSTICS BEFORE COMPLETE-CASE FILTER =====")
print("Rows in long_df:", long_df.shape[0])
print("Unique subjects in long_df:", long_df[subject_col].nunique())

diag_covars = ["age", "sex", "BrainSegVol"]
for c in diag_covars:
    if c in long_df.columns:
        print(f"Missing {c}: {long_df[c].isna().sum()}")

feature_missing_counts = long_df[pca_feature_bases].isna().sum().sort_values(ascending=False)
print("\nTop 20 missing PCA features:")
print(feature_missing_counts.head(20))

# Drop PCA features with too much missingness in long format
missing_prop = long_df[pca_feature_bases].isna().mean()
drop_threshold = 0.00   # try 0.05 or 0.10
bad_features = missing_prop[missing_prop > drop_threshold].index.tolist()

print(f"\nDropping {len(bad_features)} PCA features with > {drop_threshold*100:.1f}% missingness")
print(bad_features)

pca_feature_bases = [f for f in pca_feature_bases if f not in bad_features]
print("Remaining PCA features:", len(pca_feature_bases))
# =========================================================
# COMPLETE-CASE FILTER FOR PCA/LMM
# =========================================================
needed_cols = [subject_col, "time", "session", "BrainSegVol", "age", "sex"] + pca_feature_bases
needed_cols = [c for c in needed_cols if c in long_df.columns]

long_complete = long_df.dropna(subset=needed_cols).copy()

print("\nLong complete data:", long_complete.shape)
print("Unique subjects for PCA/LMM:", long_complete[subject_col].nunique())

# Ensure paired observations for subjects
pair_counts_before = long_df.groupby(subject_col)["time"].nunique()
print("Subjects with at least one row before complete-case filter:", (pair_counts_before >= 1).sum())

pair_counts_after = long_complete.groupby(subject_col)["time"].nunique()
paired_subjects = pair_counts_after[pair_counts_after == 2].index.tolist()
print("Paired subjects retained:", len(paired_subjects))

long_complete = long_complete[long_complete[subject_col].isin(paired_subjects)].copy()

if len(paired_subjects) < 3:
    raise ValueError("Too few paired subjects for mixed model after filtering.")

# =========================================================
# PCA ON STACKED PRE+POST
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

# Save PCA outputs
long_complete.to_csv(os.path.join(output_root, "whole_brain_long_with_pca_scores.csv"), index=False)
loadings.to_csv(os.path.join(output_root, "whole_brain_pca_loadings.csv"), index=False)
explained_var_df.to_csv(os.path.join(output_root, "whole_brain_pca_explained_variance.csv"), index=False)

# PC1 wide for delta calculations
index_cols = [c for c in [subject_col, "cohort", "id", "age", "sex", outcome_col] if c in long_complete.columns]

pc1_wide = long_complete.pivot_table(
    index=index_cols,
    columns="session",
    values="PC1"
).reset_index()

if {"sessie1", "sessie3"}.issubset(pc1_wide.columns):
    pc1_wide["PC1_delta"] = pc1_wide["sessie3"] - pc1_wide["sessie1"]
else:
    pc1_wide["PC1_delta"] = np.nan

# Add BrainSegVol delta from subject-level df
brainsegvol_delta_df = df[[subject_col, brainsegvol_pre_col, brainsegvol_post_col, brainsegvol_delta_col]].copy()
brainsegvol_delta_df = brainsegvol_delta_df.rename(columns={
    brainsegvol_pre_col: "BrainSegVol_pre",
    brainsegvol_post_col: "BrainSegVol_post",
    brainsegvol_delta_col: "BrainSegVol_delta",
})
pc1_wide = pc1_wide.merge(brainsegvol_delta_df, on=subject_col, how="left")
pc1_wide.to_csv(os.path.join(output_root, "whole_brain_pc1_subject_wide.csv"), index=False)

# =========================================================
# LINEAR MIXED-EFFECTS MODEL
# =========================================================
# Main model:
# PC1 ~ time + age + sex + BrainSegVol + (1 | subject_id)
lmm_df = long_complete[[subject_col, "PC1", "time", "age", "sex", "BrainSegVol"]].dropna().copy()

# Center continuous covariates for stability
for c in ["age", "BrainSegVol"]:
    lmm_df[f"{c}_c"] = lmm_df[c] - lmm_df[c].mean()

# sex as categorical
lmm_df["sex"] = lmm_df["sex"].astype("category")

model_formula = "PC1 ~ time + age_c + C(sex) + BrainSegVol_c"

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

lmm_summary_text = lmm_res.summary().as_text()
with open(os.path.join(output_root, "whole_brain_pc1_lmm_summary.txt"), "w") as f:
    f.write(lmm_summary_text)

lmm_params = pd.DataFrame({
    "term": lmm_res.params.index,
    "beta": lmm_res.params.values,
    "p_value": lmm_res.pvalues.reindex(lmm_res.params.index).values,
    "std_error": lmm_res.bse.reindex(lmm_res.params.index).values,
    "z_value": lmm_res.tvalues.reindex(lmm_res.params.index).values,
})
lmm_params.to_csv(os.path.join(output_root, "whole_brain_pc1_lmm_coefficients.csv"), index=False)

# =========================================================
# PC1 CHANGE vs VWREC
# =========================================================
corr_rows = []
tmp = pc1_wide[["PC1_delta", outcome_col]].dropna().copy()
if tmp.shape[0] >= 3:
    x = tmp["PC1_delta"].to_numpy(dtype=float)
    y = tmp[outcome_col].to_numpy(dtype=float)

    pear_r, pear_p = safe_pearson(x, y)
    spear_r, spear_p = safe_spearman(x, y)

    corr_rows.append({
        "metric": "PC1_delta_vs_vwrec",
        "n": len(tmp),
        "pearson_r": pear_r,
        "pearson_p": pear_p,
        "spearman_rho": spear_r,
        "spearman_p": spear_p,
    })

# optional: adjusted OLS for PC1_delta
reg_rows = []
tmp2_cols = ["PC1_delta", outcome_col, "age", "sex", "BrainSegVol_delta"]
tmp2_cols = [c for c in tmp2_cols if c in pc1_wide.columns]
tmp2 = pc1_wide[tmp2_cols].dropna().copy()

if {"PC1_delta", outcome_col, "age", "sex", "BrainSegVol_delta"}.issubset(tmp2.columns) and tmp2.shape[0] >= 5:
    tmp2["age_c"] = tmp2["age"] - tmp2["age"].mean()
    tmp2["BrainSegVol_delta_c"] = tmp2["BrainSegVol_delta"] - tmp2["BrainSegVol_delta"].mean()
    tmp2["sex"] = tmp2["sex"].astype("category")

    ols = smf.ols(
        f"PC1_delta ~ {outcome_col} + age_c + C(sex) + BrainSegVol_delta_c",
        data=tmp2
    ).fit()

    ols_summary_text = ols.summary().as_text()
    with open(os.path.join(output_root, "whole_brain_pc1_delta_vs_vwrec_ols_summary.txt"), "w") as f:
        f.write(ols_summary_text)

    reg_rows = [{
        "term": term,
        "beta": ols.params[term],
        "p_value": ols.pvalues[term],
        "std_error": ols.bse[term],
        "t_value": ols.tvalues[term],
    } for term in ols.params.index]

pd.DataFrame(corr_rows).to_csv(
    os.path.join(output_root, "whole_brain_pc1_delta_vs_vwrec_correlations.csv"),
    index=False
)
pd.DataFrame(reg_rows).to_csv(
    os.path.join(output_root, "whole_brain_pc1_delta_vs_vwrec_ols_coefficients.csv"),
    index=False
)

# =========================================================
# SAVE EXTRA DIAGNOSTICS
# =========================================================
feature_missing_counts.to_csv(
    os.path.join(output_root, "whole_brain_feature_missing_counts_long.csv"),
    header=["n_missing"]
)

subject_row_counts = long_complete.groupby(subject_col)["time"].nunique().value_counts().sort_index()
subject_row_counts.to_csv(
    os.path.join(output_root, "whole_brain_subject_pair_counts.csv"),
    header=["n_subjects"]
)

# =========================================================
# PRINT SUMMARY
# =========================================================
print("\nFinished whole-brain PCA + LMM analysis.")
print("Saved to:", output_root)

print("\nPCA explained variance:")
print(explained_var_df.to_string(index=False))

print("\nLMM coefficients:")
print(lmm_params.to_string(index=False))

if len(corr_rows) > 0:
    print("\nPC1 delta vs vwrec correlations:")
    print(pd.DataFrame(corr_rows).to_string(index=False))
else:
    print("\nNo PC1-vwrec correlation results (insufficient data).")