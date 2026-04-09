import pandas as pd
import numpy as np

# =========================
# INPUTS
# =========================
df_rad = pd.read_csv("/data/projects/CSC/code/XTC/07_regression_model/Input/x/radiomics_firstorder_shape_hippocampus_thalamus.csv")  # radiomics features
df_y   = pd.read_csv("/data/projects/CSC/code/XTC/07_regression_model/Input/y/XTC_naief_npo.csv", sep = ';')  # cognitive scores

pre_session = 1
post_session = 3
outcome_col = "vwrec"   # 15 words test recall
covariates = ["geslacht", "age"]

# =========================
# 1. Parse subject_id
# =========================
rad = df_rad.copy()

parsed = rad["subject_id"].str.extract(r"^(I{1,3}|IV)(\d+)_sessie(\d+)$")
parsed.columns = ["cohort_roman", "id", "session"]

cohort_map = {"I": 1, "II": 2, "III": 3, "IV": 4}

rad["cohort"] = parsed["cohort_roman"].map(cohort_map)
rad["id"] = parsed["id"].astype(int)
rad["session"] = parsed["session"].astype(int)

# =========================
# 2. Select feature cols
# =========================
id_cols = ["subject_id", "roi_name", "cohort", "id", "session", "cohort_roman"]
feature_cols = [c for c in rad.columns if c not in id_cols]

# =========================
# 3. Pivot ROIs wide
# =========================
rad_wide = rad.pivot_table(
    index=["cohort", "id", "session"],
    columns="roi_name",
    values=feature_cols,
    aggfunc="first"
)

rad_wide.columns = [f"{roi}_{feat}" for feat, roi in rad_wide.columns]
rad_wide = rad_wide.reset_index()

# =========================
# 4. Split pre/post
# =========================
rad_pre = rad_wide[rad_wide["session"] == pre_session].copy().drop(columns="session")
rad_post = rad_wide[rad_wide["session"] == post_session].copy().drop(columns="session")

rad_cols = [c for c in rad_pre.columns if c not in ["cohort", "id"]]

rad_pre = rad_pre.rename(columns={c: f"{c}_pre" for c in rad_cols})
rad_post = rad_post.rename(columns={c: f"{c}_post" for c in rad_cols})

# =========================
# 5. Pair pre and post
# =========================
rad_pair = rad_pre.merge(rad_post, on=["cohort", "id"], how="inner")

# =========================
# 6. Compute delta
# =========================
# Keep only actual radiomics columns
rad_feature_cols = [
    c for c in rad_wide.columns
    if c not in ["cohort", "id", "session"]
    and not c.endswith("image_path")
    and not c.endswith("roi_label")
    and (
        "firstorder" in c
        or "shape" in c
    )
]

# Build a clean wide table
rad_wide_clean = rad_wide[["cohort", "id", "session"] + rad_feature_cols].copy()
# Split pre/post
rad_pre = rad_wide_clean[rad_wide_clean["session"] == pre_session].copy().drop(columns="session")
rad_post = rad_wide_clean[rad_wide_clean["session"] == post_session].copy().drop(columns="session")

rad_cols = [c for c in rad_pre.columns if c not in ["cohort", "id"]]

# Convert to numeric just to be safe
for c in rad_cols:
    rad_pre[c] = pd.to_numeric(rad_pre[c], errors="coerce")
    rad_post[c] = pd.to_numeric(rad_post[c], errors="coerce")

# Rename
rad_pre = rad_pre.rename(columns={c: f"{c}_pre" for c in rad_cols})
rad_post = rad_post.rename(columns={c: f"{c}_post" for c in rad_cols})

# Merge
rad_pair = rad_pre.merge(rad_post, on=["cohort", "id"], how="inner")

# Delta
for c in rad_cols:
    rad_pair[f"{c}_delta"] = rad_pair[f"{c}_post"] - rad_pair[f"{c}_pre"]

# =========================
# 7. Prepare outcome table
# =========================
y_df = df_y.copy()
y_df["cohort"] = 1
y_df["id"] = y_df["id"].astype(int)

# =========================
# 8. Merge X with y
# =========================
model_df = rad_pair.merge(y_df, on=["cohort", "id"], how="inner")
# =========================
# 9. Build predictor sets
# =========================
pre_cols = [c for c in model_df.columns if c.endswith("_pre")]
delta_cols = [c for c in model_df.columns if c.endswith("_delta")]

X_pre = model_df[pre_cols + covariates]
X_delta = model_df[delta_cols + covariates]
X_pre_delta = model_df[pre_cols + delta_cols + covariates]

y = model_df[outcome_col]

# =========================
# 10. Drop missing
# =========================
keep = ~y.isna()
X_pre = X_pre.loc[keep].copy()
X_delta = X_delta.loc[keep].copy()
X_pre_delta = X_pre_delta.loc[keep].copy()
y = y.loc[keep].copy()

complete = ~X_pre_delta.isna().any(axis=1)
X_pre = X_pre.loc[complete].copy()
X_delta = X_delta.loc[complete].copy()
X_pre_delta = X_pre_delta.loc[complete].copy()
y = y.loc[complete].copy()

print("N subjects in final dataset:", len(y))
print("X_pre shape:", X_pre.shape)
print("X_delta shape:", X_delta.shape)
print("X_pre_delta shape:", X_pre_delta.shape)

# Save final datasets
id_cols = ["cohort", "id"]

pre_cols = [c for c in model_df.columns if c.endswith("_pre")]
delta_cols = [c for c in model_df.columns if c.endswith("_delta")]

X_pre = model_df[id_cols + pre_cols + covariates].copy()
X_delta = model_df[id_cols + delta_cols + covariates].copy()
X_pre_delta = model_df[id_cols + pre_cols + delta_cols + covariates].copy()

y_df_final = model_df[id_cols + [outcome_col]].copy()
out_dir = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed"

X_pre.to_csv(f"{out_dir}/X_pre_with_id.csv", index=False)
X_delta.to_csv(f"{out_dir}/X_delta_with_id.csv", index=False)
X_pre_delta.to_csv(f"{out_dir}/X_pre_delta_with_id.csv", index=False)
y_df_final.to_csv(f"{out_dir}/y_with_id.csv", index=False)

print("Saved X and y with cohort/id.")

