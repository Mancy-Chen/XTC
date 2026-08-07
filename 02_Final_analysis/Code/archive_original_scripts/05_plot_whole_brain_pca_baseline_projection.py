import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# =========================================================
# SETTINGS
# =========================================================
output_root = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_lmm_with_xtc_dose_baseline_projection"

long_path = os.path.join(
    output_root,
    "whole_brain_long_with_pca_scores_and_dose_baseline_projection.csv"
)

loadings_path = os.path.join(
    output_root,
    "whole_brain_pca_loadings_baseline_fit.csv"
)

subject_col = "subject_id"
dose_col = "xlttot_sessie3"
use_log_dose = True

fig1_path = os.path.join(
    output_root,
    "lmm_time_effect_across_continuous_xtc_dose_all_subjects_95CI.png"
)

fig2_path = os.path.join(
    output_root,
    "pc1_loadings_plot.png"
)

fig3_path = os.path.join(
    output_root,
    "lmm_time_effect_across_continuous_xtc_dose_xtc_users_only_95CI.png"
)

os.makedirs(output_root, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
long_df = pd.read_csv(long_path)

# Basic numeric conversion
long_df["PC1"] = pd.to_numeric(long_df["PC1"], errors="coerce")
long_df["time"] = pd.to_numeric(long_df["time"], errors="coerce")
long_df["age"] = pd.to_numeric(long_df["age"], errors="coerce")
long_df[dose_col] = pd.to_numeric(long_df[dose_col], errors="coerce")

# Support either BrainSegVol or the original FastSurfer column name
if "BrainSegVol" not in long_df.columns:
    if "aseg+DKT_BrainSegVol" in long_df.columns:
        long_df["BrainSegVol"] = long_df["aseg+DKT_BrainSegVol"]
    else:
        raise ValueError(
            "Could not find BrainSegVol column. "
            "Expected either 'BrainSegVol' or 'aseg+DKT_BrainSegVol'."
        )

long_df["BrainSegVol"] = pd.to_numeric(long_df["BrainSegVol"], errors="coerce")

# Dose transformation
if use_log_dose:
    long_df["dose_xtc"] = np.log1p(long_df[dose_col])
    x_label = "log(1 + total XTC dose)"
else:
    long_df["dose_xtc"] = long_df[dose_col]
    x_label = "Total XTC dose"


# =========================================================
# PREPARE MODEL DATA
# =========================================================
model_df = long_df[
    [subject_col, "PC1", "time", "dose_xtc", "age", "sex", "BrainSegVol"]
].dropna().copy()

# Center covariates using full model sample
model_df["dose_xtc_c"] = model_df["dose_xtc"] - model_df["dose_xtc"].mean()
model_df["age_c"] = model_df["age"] - model_df["age"].mean()
model_df["BrainSegVol_c"] = model_df["BrainSegVol"] - model_df["BrainSegVol"].mean()
model_df["sex"] = model_df["sex"].astype("category")

formula = "PC1 ~ time * dose_xtc_c + age_c + C(sex) + BrainSegVol_c"


# =========================================================
# FIT FULL PRIMARY LMM
# =========================================================
try:
    model = smf.mixedlm(
        formula,
        data=model_df,
        groups=model_df[subject_col],
        re_formula="1"
    )
    res = model.fit(reml=False, method="lbfgs")
except Exception:
    model = smf.mixedlm(
        formula,
        data=model_df,
        groups=model_df[subject_col]
    )
    res = model.fit(reml=False)

print(res.summary())


# =========================================================
# SUBJECT-LEVEL OBSERVED PC1 CHANGE
# =========================================================
wide_pc1 = model_df.pivot_table(
    index=subject_col,
    columns="time",
    values="PC1"
)

dose_subject = model_df.groupby(subject_col)["dose_xtc"].first()
raw_dose_subject = long_df.groupby(subject_col)[dose_col].first()

obs_df = wide_pc1.copy()
obs_df["dose_xtc"] = dose_subject
obs_df[dose_col] = raw_dose_subject

if 0 not in obs_df.columns or 1 not in obs_df.columns:
    raise ValueError(
        "Expected time values 0 and 1 in the long PCA-score file. "
        "Please check how baseline and follow-up are coded."
    )

obs_df["PC1_delta"] = obs_df[1] - obs_df[0]
obs_df = obs_df.dropna(subset=["dose_xtc", "PC1_delta"])


# =========================================================
# EXTRACT LMM TERMS FOR PREDICTION AND CI
# =========================================================
params = res.params
cov = res.cov_params()

time_term = "time"

possible_interaction_terms = [
    "time:dose_xtc_c",
    "dose_xtc_c:time"
]

interaction_term = None
for term in possible_interaction_terms:
    if term in params.index:
        interaction_term = term
        break

if interaction_term is None:
    raise ValueError("Could not find time × dose interaction term in model parameters.")

print("\nKey fixed effects:")
print(params[[time_term, "dose_xtc_c", interaction_term]])

print("\nFixed-effect covariance used for CI:")
print(cov.loc[[time_term, interaction_term], [time_term, interaction_term]])


# =========================================================
# HELPER FUNCTION: LMM-ESTIMATED CHANGE AND 95% CI
# =========================================================
def estimate_lmm_change_with_ci(dose_grid, model_df, params, cov, time_term, interaction_term):
    """
    Estimate LMM-predicted PC1 change from baseline to follow-up:

        ΔPC1 = beta_time + beta_time:dose * centered_dose

    The 95% CI is based on the covariance between beta_time and
    beta_time:dose.
    """

    dose_grid_c = dose_grid - model_df["dose_xtc"].mean()

    est_change = (
        params[time_term]
        + params[interaction_term] * dose_grid_c
    )

    var_change = (
        cov.loc[time_term, time_term]
        + (dose_grid_c ** 2) * cov.loc[interaction_term, interaction_term]
        + 2 * dose_grid_c * cov.loc[time_term, interaction_term]
    )

    se_change = np.sqrt(np.maximum(var_change, 0))

    ci_low = est_change - 1.96 * se_change
    ci_high = est_change + 1.96 * se_change

    return est_change, ci_low, ci_high


# =========================================================
# FIGURE 1: ALL SUBJECTS — LMM-ESTIMATED PC1 CHANGE
# =========================================================
dose_grid_all = np.linspace(
    obs_df["dose_xtc"].min(),
    obs_df["dose_xtc"].max(),
    200
)

est_change_all, ci_low_all, ci_high_all = estimate_lmm_change_with_ci(
    dose_grid=dose_grid_all,
    model_df=model_df,
    params=params,
    cov=cov,
    time_term=time_term,
    interaction_term=interaction_term
)

plt.figure(figsize=(10, 7))

plt.scatter(
    obs_df["dose_xtc"],
    obs_df["PC1_delta"],
    alpha=0.35,
    s=55,
    label=f"Observed subject-level change, n={len(obs_df)}"
)

plt.plot(
    dose_grid_all,
    est_change_all,
    linewidth=3,
    label="LMM-estimated change"
)

plt.fill_between(
    dose_grid_all,
    ci_low_all,
    ci_high_all,
    alpha=0.20,
    label="95% CI"
)

plt.axhline(0, linestyle="--", linewidth=1.5)

plt.title("LMM-estimated PC1 change across continuous XTC dose", fontsize=16)
plt.xlabel(x_label, fontsize=13)
plt.ylabel("Estimated change in whole-brain PC1\n(follow-up − baseline)", fontsize=13)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(fig1_path, dpi=300)
plt.close()

print(f"Saved Figure 1 to: {fig1_path}")


# =========================================================
# FIGURE 2: TOP 20 PC1 LOADINGS
# =========================================================
loadings_df = pd.read_csv(loadings_path)

# Supports both possible formats:
#   feature, PC1, PC2, ...
#   feature, loading_PC1
if "PC1" in loadings_df.columns:
    loading_col = "PC1"
elif "loading_PC1" in loadings_df.columns:
    loading_col = "loading_PC1"
else:
    raise ValueError(
        f"Could not find PC1 loading column. "
        f"Available columns: {loadings_df.columns.tolist()}"
    )

if "feature" not in loadings_df.columns:
    raise ValueError(
        f"Could not find 'feature' column in loadings file. "
        f"Available columns: {loadings_df.columns.tolist()}"
    )

loadings_df[loading_col] = pd.to_numeric(loadings_df[loading_col], errors="coerce")
loadings_df["abs_PC1"] = loadings_df[loading_col].abs()

top20 = (
    loadings_df
    .dropna(subset=[loading_col])
    .sort_values("abs_PC1", ascending=False)
    .head(20)
    .sort_values(loading_col, ascending=True)
)

plt.figure(figsize=(10, 8))

plt.barh(
    top20["feature"],
    top20[loading_col]
)

plt.title("Top 20 loadings for PC1", fontsize=14)
plt.xlabel("PC1 loading", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()

plt.savefig(fig2_path, dpi=300)
plt.close()

print(f"Saved Figure 2 to: {fig2_path}")


# =========================================================
# FIGURE 3: XTC USERS ONLY — LMM-ESTIMATED PC1 CHANGE
# =========================================================
# This figure removes XTC-naive participants from the scatter plot.
# The fitted line and 95% CI still come from the full primary LMM.

obs_user_df = obs_df[obs_df[dose_col] > 0].copy()

if obs_user_df.empty:
    raise ValueError("No XTC users found. Check dose coding and dose_col.")

dose_grid_user = np.linspace(
    obs_user_df["dose_xtc"].min(),
    obs_user_df["dose_xtc"].max(),
    200
)

est_change_user, ci_low_user, ci_high_user = estimate_lmm_change_with_ci(
    dose_grid=dose_grid_user,
    model_df=model_df,
    params=params,
    cov=cov,
    time_term=time_term,
    interaction_term=interaction_term
)

plt.figure(figsize=(10, 7))

plt.scatter(
    obs_user_df["dose_xtc"],
    obs_user_df["PC1_delta"],
    alpha=0.65,
    s=60,
    label=f"Observed XTC users only, n={len(obs_user_df)}"
)

plt.plot(
    dose_grid_user,
    est_change_user,
    linewidth=3,
    label="LMM-estimated change"
)

plt.fill_between(
    dose_grid_user,
    ci_low_user,
    ci_high_user,
    alpha=0.20,
    label="95% CI"
)

plt.axhline(0, linestyle="--", linewidth=1.5)

plt.title(
    "LMM-estimated PC1 change across XTC dose among XTC users",
    fontsize=16
)
plt.xlabel(x_label, fontsize=13)
plt.ylabel("Estimated change in whole-brain PC1\n(follow-up − baseline)", fontsize=13)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(fig3_path, dpi=300)
plt.close()

print(f"Saved Figure 3 to: {fig3_path}")


# =========================================================
# DONE
# =========================================================
print("\nAll figures saved:")
print(f"Figure 1: {fig1_path}")
print(f"Figure 2: {fig2_path}")
print(f"Figure 3: {fig3_path}")