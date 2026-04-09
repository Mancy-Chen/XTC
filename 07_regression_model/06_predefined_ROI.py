import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# =========================================================
# PATH
# =========================================================
data_path = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_lmm_with_xtc_dose/whole_brain_long_with_pca_scores_and_dose.csv"
out_path = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_lmm_with_xtc_dose/roi_targeted_lmm_results.csv"

# =========================================================
# LOAD
# =========================================================
df = pd.read_csv(data_path)

# =========================================================
# SETTINGS
# =========================================================
subject_col = "subject_id"
dose_col = "xlttot_sessie3"
use_log_dose = True

# targeted ROIs
target_rois = [
    "aseg+DKT_Left-Hippocampus",
    "aseg+DKT_Right-Hippocampus",
    "aseg+DKT_Left-Thalamus",
    "aseg+DKT_Right-Thalamus",
    "aseg+DKT_Left-Cerebral-White-Matter",
    "aseg+DKT_Right-Cerebral-White-Matter",
    "aseg+DKT_Left-VentralDC",
    "aseg+DKT_Right-VentralDC",
    "aseg+DKT_Brain-Stem",
    "aseg+DKT_ctx-lh-insula",
    "aseg+DKT_ctx-rh-insula",
    "aseg+DKT_ctx-lh-superiorfrontal",
    "aseg+DKT_ctx-rh-superiorfrontal",
]

# keep only ROIs that actually exist
target_rois = [r for r in target_rois if r in df.columns]
print("Testing ROIs:")
print(target_rois)

# =========================================================
# PREP
# =========================================================
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["BrainSegVol"] = pd.to_numeric(df["BrainSegVol"], errors="coerce")
df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce")
df["sex"] = df["sex"].astype("category")

if use_log_dose:
    df["dose_xtc"] = np.log1p(df[dose_col])
else:
    df["dose_xtc"] = df[dose_col]

df["dose_xtc_c"] = df["dose_xtc"] - df["dose_xtc"].mean()
df["age_c"] = df["age"] - df["age"].mean()
df["BrainSegVol_c"] = df["BrainSegVol"] - df["BrainSegVol"].mean()

# =========================================================
# RUN TARGETED ROI MODELS
# =========================================================
results = []

for roi in target_rois:
    tmp = df[[subject_col, roi, "time", "dose_xtc_c", "age_c", "sex", "BrainSegVol_c"]].dropna().copy()

    if tmp[subject_col].nunique() < 10:
        print(f"Skipping {roi}: too few subjects")
        continue

    tmp = tmp.rename(columns={roi: "ROI_value"})

    try:
        model = smf.mixedlm(
            "ROI_value ~ time * dose_xtc_c + age_c + C(sex) + BrainSegVol_c",
            data=tmp,
            groups=tmp[subject_col],
            re_formula="1"
        )
        res = model.fit(reml=False, method="lbfgs")

        results.append({
            "ROI": roi,
            "n_subjects": tmp[subject_col].nunique(),
            "n_rows": tmp.shape[0],
            "beta_time": res.params.get("time", np.nan),
            "p_time": res.pvalues.get("time", np.nan),
            "beta_dose": res.params.get("dose_xtc_c", np.nan),
            "p_dose": res.pvalues.get("dose_xtc_c", np.nan),
            "beta_time_dose": res.params.get("time:dose_xtc_c", np.nan),
            "p_time_dose": res.pvalues.get("time:dose_xtc_c", np.nan),
            "beta_age": res.params.get("age_c", np.nan),
            "p_age": res.pvalues.get("age_c", np.nan),
            "beta_brainseg": res.params.get("BrainSegVol_c", np.nan),
            "p_brainseg": res.pvalues.get("BrainSegVol_c", np.nan),
        })

        print(f"Finished {roi}")

    except Exception as e:
        print(f"Failed {roi}: {e}")

# =========================================================
# SAVE RESULTS
# =========================================================
res_df = pd.DataFrame(results)

if not res_df.empty:
    mask = res_df["p_time_dose"].notna()
    qvals = np.full(len(res_df), np.nan)
    if mask.sum() > 0:
        _, q, _, _ = multipletests(res_df.loc[mask, "p_time_dose"], method="fdr_bh")
        qvals[mask] = q
    res_df["q_time_dose_fdr"] = qvals

    res_df = res_df.sort_values(["q_time_dose_fdr", "p_time_dose"], ascending=[True, True]).reset_index(drop=True)

print("\nTargeted ROI results:")
print(res_df)

res_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")