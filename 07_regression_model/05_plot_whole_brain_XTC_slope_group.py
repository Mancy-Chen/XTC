import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import norm

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
dose_col = "xlttot_sessie3"   # change if needed
use_log_dose = True

# =========================================================
# PREP
# =========================================================
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["sex"] = df["sex"].astype("category")
df["BrainSegVol"] = pd.to_numeric(df["BrainSegVol"], errors="coerce")
df["PC1"] = pd.to_numeric(df["PC1"], errors="coerce")
df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce")

if use_log_dose:
    df["dose_xtc"] = np.log1p(df[dose_col])
else:
    df["dose_xtc"] = df[dose_col]

# ---------------------------------------------------------
# Make dose groups:
# Zero dose / Low non-zero / High non-zero
# ---------------------------------------------------------
df["dose_group"] = np.nan

zero_mask = df["dose_xtc"] == 0
nonzero_mask = df["dose_xtc"] > 0

df.loc[zero_mask, "dose_group"] = "Zero dose"

nonzero_df = df.loc[nonzero_mask].copy()

# split non-zero group into two halves by median
median_nonzero = nonzero_df["dose_xtc"].median()

df.loc[nonzero_mask & (df["dose_xtc"] <= median_nonzero), "dose_group"] = "Low non-zero dose"
df.loc[nonzero_mask & (df["dose_xtc"] > median_nonzero), "dose_group"] = "High non-zero dose"

# set category order explicitly
df["dose_group"] = pd.Categorical(
    df["dose_group"],
    categories=["Zero dose", "Low non-zero dose", "High non-zero dose"],
    ordered=True
)

print("\nDose group counts:")
print(df["dose_group"].value_counts(dropna=False))

# center continuous covariates
df["age_c"] = df["age"] - df["age"].mean()
df["BrainSegVol_c"] = df["BrainSegVol"] - df["BrainSegVol"].mean()

# keep complete rows for the model
model_df = df[
    [subject_col, "PC1", "time", "dose_group", "age_c", "sex", "BrainSegVol_c"]
].dropna().copy()

print("\nModel df:", model_df.shape)
print("Unique subjects:", model_df[subject_col].nunique())
print("\nDose group counts in model_df:")
print(model_df["dose_group"].value_counts())

# =========================================================
# FIT LMM
# =========================================================
model = smf.mixedlm(
    "PC1 ~ time * C(dose_group) + age_c + C(sex) + BrainSegVol_c",
    data=model_df,
    groups=model_df[subject_col],
    re_formula="1"
)

result = model.fit(reml=False, method="lbfgs")
print(result.summary())

# =========================================================
# SIMPLE SLOPES FOR TIME WITHIN EACH DOSE GROUP
# =========================================================
b = result.params
cov = result.cov_params()

print("\n===== SIMPLE SLOPES FOR TIME BY DOSE GROUP =====")

groups = ["Zero dose", "Low non-zero dose", "High non-zero dose"]
rows = []

# reference group slope = coefficient of time
ref_group = model_df["dose_group"].cat.categories[0]
time_coef_name = "time"

for group in groups:
    if group == ref_group:
        slope = b[time_coef_name]
        var_slope = cov.loc[time_coef_name, time_coef_name]
    else:
        int_name = f"time:C(dose_group)[T.{group}]"

        slope = b[time_coef_name] + b[int_name]

        var_slope = (
            cov.loc[time_coef_name, time_coef_name]
            + cov.loc[int_name, int_name]
            + 2 * cov.loc[time_coef_name, int_name]
        )

    se = np.sqrt(var_slope)
    z = slope / se
    p = 2 * (1 - norm.cdf(abs(z)))

    rows.append({
        "dose_group": group,
        "time_slope": slope,
        "SE": se,
        "z": z,
        "p": p
    })

simple_slopes_df = pd.DataFrame(rows)
print(simple_slopes_df)

simple_slopes_df.to_csv("simple_slopes_time_by_dose_group.csv", index=False)
print("\nSaved: simple_slopes_time_by_dose_group.csv")

# =========================================================
# PAIRWISE DIFFERENCES BETWEEN TIME SLOPES
# =========================================================
print("\n===== PAIRWISE DIFFERENCES BETWEEN TIME SLOPES =====")

pairwise_rows = []

def get_slope_info(group, b, cov, ref_group="Zero dose"):
    """
    Returns:
        slope_expr coefficients as dict over model params
    """
    expr = {"time": 1.0}

    if group != ref_group:
        expr[f"time:C(dose_group)[T.{group}]"] = 1.0

    return expr

def linear_combo_stats(expr, b, cov):
    """
    expr: dict of {param_name: weight}
    """
    params = list(expr.keys())
    weights = np.array([expr[p] for p in params])

    est = sum(weights[i] * b[params[i]] for i in range(len(params)))

    var = 0.0
    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            var += weights[i] * weights[j] * cov.loc[pi, pj]

    se = np.sqrt(var)
    z = est / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return est, se, z, p

comparisons = [
    ("Zero dose", "Low non-zero dose"),
    ("Zero dose", "High non-zero dose"),
    ("Low non-zero dose", "High non-zero dose"),
]

for g1, g2 in comparisons:
    expr1 = get_slope_info(g1, b, cov, ref_group=ref_group)
    expr2 = get_slope_info(g2, b, cov, ref_group=ref_group)

    # difference = slope_g2 - slope_g1
    all_params = set(expr1.keys()).union(expr2.keys())
    diff_expr = {p: expr2.get(p, 0.0) - expr1.get(p, 0.0) for p in all_params}

    diff, se, z, p = linear_combo_stats(diff_expr, b, cov)

    pairwise_rows.append({
        "comparison": f"{g2} - {g1}",
        "slope_difference": diff,
        "SE": se,
        "z": z,
        "p": p
    })

comp_df = pd.DataFrame(pairwise_rows)
print(comp_df)

comp_df.to_csv("dose_group_slope_differences.csv", index=False)
print("\nSaved: dose_group_slope_differences.csv")

# =========================================================
# INTERACTION PLOT
# =========================================================
# Predicted PC1 at Session 1 and Session 3 for each dose group
# Hold age_c = 0, BrainSegVol_c = 0, sex = reference category
plot_rows = []

ref_sex = model_df["sex"].cat.categories[0]

for group in groups:
    for time_value in [0, 1]:
        pred_df = pd.DataFrame({
            "time": [time_value],
            "dose_group": pd.Categorical([group], categories=model_df["dose_group"].cat.categories),
            "age_c": [0],
            "BrainSegVol_c": [0],
            "sex": pd.Categorical([ref_sex], categories=model_df["sex"].cat.categories)
        })

        pred = result.predict(pred_df)[0]

        plot_rows.append({
            "dose_group": group,
            "time": time_value,
            "predicted_PC1": pred
        })

plot_df = pd.DataFrame(plot_rows)
plot_df["session"] = plot_df["time"].map({0: "Session 1", 1: "Session 3"})

print("\nPredicted values for interaction plot:")
print(plot_df)

# plot
plt.figure(figsize=(8, 6))

for group in groups:
    sub = plot_df[plot_df["dose_group"] == group].sort_values("time")
    plt.plot(sub["time"], sub["predicted_PC1"], marker="o", linewidth=2, label=group)

plt.xticks([0, 1], ["Baseline", "Follow-up"])
plt.ylabel("Predicted PC1")
plt.xlabel("Time")
plt.title("Interaction of XTC dose group and time on whole-brain PC1")
plt.legend()
plt.tight_layout()
plt.savefig("interaction_plot_time_by_dose_group.png", dpi=300, bbox_inches="tight")
# plt.show()

print("\nSaved: interaction_plot_time_by_dose_group.png")