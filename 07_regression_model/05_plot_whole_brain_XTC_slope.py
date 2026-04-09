import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.formula.api as smf

# =========================================================
# PATH
# =========================================================
data_path = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_lmm_with_xtc_dose/whole_brain_long_with_pca_scores_and_dose.csv"

# =========================================================
# LOAD
# =========================================================
df = pd.read_csv(data_path)

print(df.shape)
print(df.columns.tolist())

# =========================================================
# SETTINGS
# =========================================================
subject_col = "subject_id"
dose_col = "xlttot_sessie3"
use_log_dose = True

# =========================================================
# PREP
# =========================================================
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["BrainSegVol"] = pd.to_numeric(df["BrainSegVol"], errors="coerce")
df["PC1"] = pd.to_numeric(df["PC1"], errors="coerce")
df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce")
df["sex"] = df["sex"].astype("category")

if use_log_dose:
    df["dose_xtc"] = np.log1p(df[dose_col])
    x_label = f"log(1 + {dose_col})"
else:
    df["dose_xtc"] = df[dose_col]
    x_label = dose_col

# center continuous covariates for the model
df["dose_xtc_c"] = df["dose_xtc"] - df["dose_xtc"].mean()
df["age_c"] = df["age"] - df["age"].mean()
df["BrainSegVol_c"] = df["BrainSegVol"] - df["BrainSegVol"].mean()

# keep complete rows for the model
model_df = df[
    [subject_col, "PC1", "time", "dose_xtc", "dose_xtc_c", "age_c", "sex", "BrainSegVol_c"]
].dropna().copy()

print("Model df:", model_df.shape)
print("Unique subjects:", model_df[subject_col].nunique())

# =========================================================
# DEFINE DOSAGE GROUPS
# =========================================================
# Zero dose / Low non-zero / High non-zero
model_df["dose_group"] = np.nan

zero_mask = model_df["dose_xtc"] == 0
nonzero_mask = model_df["dose_xtc"] > 0

model_df.loc[zero_mask, "dose_group"] = "Zero dose"

nonzero_vals = model_df.loc[nonzero_mask, "dose_xtc"].dropna()
median_nonzero = nonzero_vals.median()

model_df.loc[nonzero_mask & (model_df["dose_xtc"] <= median_nonzero), "dose_group"] = "Low non-zero dose"
model_df.loc[nonzero_mask & (model_df["dose_xtc"] > median_nonzero), "dose_group"] = "High non-zero dose"

print("\nDose group counts:")
print(model_df["dose_group"].value_counts(dropna=False))
print("\nMedian non-zero dose_xtc:", median_nonzero)

# representative dose values for each group
group_representatives = {
    "Zero dose": 0.0,
    "Low non-zero dose": model_df.loc[model_df["dose_group"] == "Low non-zero dose", "dose_xtc"].median(),
    "High non-zero dose": model_df.loc[model_df["dose_group"] == "High non-zero dose", "dose_xtc"].median(),
}

print("\nRepresentative dose values:")
for k, v in group_representatives.items():
    print(k, "->", v)

# =========================================================
# FIT LMM
# =========================================================
model = smf.mixedlm(
    "PC1 ~ time * dose_xtc_c + age_c + C(sex) + BrainSegVol_c",
    data=model_df,
    groups=model_df[subject_col],
    re_formula="1"
)

result = model.fit(reml=False, method="lbfgs")
print(result.summary())

b = result.params
cov = result.cov_params()
dose_mean = model_df["dose_xtc"].mean()

# =========================================================
# SIMPLE SLOPES FOR TIME AT THE THREE DOSAGE GROUPS
# =========================================================
print("\n===== SIMPLE SLOPES FOR TIME BY DOSAGE GROUP =====")
simple_slopes = []

for label, dose_value in group_representatives.items():
    dose_c = dose_value - dose_mean

    # slope of time at this dose
    slope = b["time"] + dose_c * b["time:dose_xtc_c"]

    # variance of slope
    var_slope = (
        cov.loc["time", "time"]
        + (dose_c ** 2) * cov.loc["time:dose_xtc_c", "time:dose_xtc_c"]
        + 2 * dose_c * cov.loc["time", "time:dose_xtc_c"]
    )

    se_slope = np.sqrt(var_slope)
    z_value = slope / se_slope
    p_value = 2 * (1 - norm.cdf(abs(z_value)))

    simple_slopes.append({
        "dose_group": label,
        "representative_dose_xtc": dose_value,
        "dose_xtc_c": dose_c,
        "time_slope": slope,
        "SE": se_slope,
        "z": z_value,
        "p": p_value
    })

simple_slopes_df = pd.DataFrame(simple_slopes)
print(simple_slopes_df)

simple_slopes_df.to_csv("simple_slopes_time_by_dose_group.csv", index=False)
print("\nSaved: simple_slopes_time_by_dose_group.csv")

# =========================================================
# INTERACTION PLOT USING THE THREE DOSAGE GROUPS
# =========================================================
plot_rows = []

for label, dose_value in group_representatives.items():
    dose_c = dose_value - dose_mean
    for time_value in [0, 1]:
        pred_df = pd.DataFrame({
            "time": [time_value],
            "dose_xtc_c": [dose_c],
            "age_c": [0],
            "BrainSegVol_c": [0],
            "sex": [model_df["sex"].cat.categories[0]]
        })

        pred = result.predict(pred_df)[0]

        plot_rows.append({
            "dose_group": label,
            "time": time_value,
            "predicted_PC1": pred
        })

plot_df = pd.DataFrame(plot_rows)
plot_df["session"] = plot_df["time"].map({0: "Session 1", 1: "Session 3"})

print("\nPredicted values for interaction plot:")
print(plot_df)

plt.figure(figsize=(8, 6))

group_order = ["Zero dose", "Low non-zero dose", "High non-zero dose"]

for group in group_order:
    sub = plot_df[plot_df["dose_group"] == group].sort_values("time")
    plt.plot(sub["time"], sub["predicted_PC1"], marker="o", linewidth=2, label=group)

plt.xticks([0, 1], ["Baseline", "Follow-up"])
plt.ylabel("Predicted PC1")
plt.xlabel("Time")
plt.title("Interaction of XTC dose group and time on whole-brain PC1")
plt.legend()
plt.tight_layout()
plt.savefig("interaction_plot_time_by_dose_group.png", dpi=300, bbox_inches="tight")
print("\nSaved: interaction_plot_time_by_dose_group.png")

# =========================================================
# PAIRWISE SLOPE DIFFERENCE TESTS BETWEEN GROUPS
# =========================================================
comparisons = [
    ("Zero vs Low non-zero", "Zero dose", "Low non-zero dose"),
    ("Low non-zero vs High non-zero", "Low non-zero dose", "High non-zero dose"),
    ("Zero vs High non-zero", "Zero dose", "High non-zero dose"),
]

rows = []
beta_int = result.params["time:dose_xtc_c"]
var_int = cov.loc["time:dose_xtc_c", "time:dose_xtc_c"]

for label, g1, g2 in comparisons:
    d1 = group_representatives[g1] - dose_mean
    d2 = group_representatives[g2] - dose_mean

    diff = beta_int * (d2 - d1)
    se = np.sqrt(var_int * (d2 - d1) ** 2)
    z = diff / se
    p = 2 * (1 - norm.cdf(abs(z)))

    rows.append({
        "comparison": label,
        "group1": g1,
        "group2": g2,
        "slope_difference": diff,
        "SE": se,
        "z": z,
        "p": p
    })

comp_df = pd.DataFrame(rows)
print("\nDifferences between time slopes:")
print(comp_df)

comp_df.to_csv("dose_group_slope_differences.csv", index=False)
print("\nSaved: dose_group_slope_differences.csv")