import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------------------------------------
# Load subject-level file
# ---------------------------------------------------------
data_path = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_lmm_with_xtc_dose/whole_brain_pc1_subject_wide_with_dose.csv"
df = pd.read_csv(data_path)

dose_col = "xlttot_sessie3"
use_log_dose = True

# ---------------------------------------------------------
# Prepare
# ---------------------------------------------------------
df["PC1_delta"] = pd.to_numeric(df["PC1_delta"], errors="coerce")
df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce")

if use_log_dose:
    df["dose_xtc"] = np.log1p(df[dose_col])
    x_label = f"log(1 + {dose_col})"
else:
    df["dose_xtc"] = df[dose_col]
    x_label = dose_col

plot_df = df[["dose_xtc", "PC1_delta"]].dropna().copy()

# ---------------------------------------------------------
# Make dose groups:
# Zero dose / Low non-zero / High non-zero
# ---------------------------------------------------------
plot_df["dose_group"] = np.nan

zero_mask = plot_df["dose_xtc"] == 0
nonzero_mask = plot_df["dose_xtc"] > 0

plot_df.loc[zero_mask, "dose_group"] = "Zero dose"

nonzero_df = plot_df.loc[nonzero_mask].copy()

# split non-zero group into two halves by median
median_nonzero = nonzero_df["dose_xtc"].median()

plot_df.loc[nonzero_mask & (plot_df["dose_xtc"] <= median_nonzero), "dose_group"] = "Low non-zero dose"
plot_df.loc[nonzero_mask & (plot_df["dose_xtc"] > median_nonzero), "dose_group"] = "High non-zero dose"

print(plot_df["dose_group"].value_counts(dropna=False))

# ---------------------------------------------------------
# Plot scatter + separate regression lines
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

group_order = ["Zero dose", "Low non-zero dose", "High non-zero dose"]

for group in group_order:
    sub = plot_df[plot_df["dose_group"] == group].copy()

    if sub.empty:
        continue

    # scatter points
    plt.scatter(sub["dose_xtc"], sub["PC1_delta"], alpha=0.7, label=group)

    # fit simple OLS line only if x has >1 unique value
    if sub["dose_xtc"].nunique() > 1:
        X = sm.add_constant(sub["dose_xtc"])
        y = sub["PC1_delta"]
        model = sm.OLS(y, X).fit()

        x_grid = np.linspace(sub["dose_xtc"].min(), sub["dose_xtc"].max(), 100)
        X_grid = sm.add_constant(x_grid)
        pred = model.get_prediction(X_grid).summary_frame(alpha=0.05)

        plt.plot(x_grid, pred["mean"], linewidth=2)
        plt.fill_between(
            x_grid,
            pred["mean_ci_lower"].to_numpy(),
            pred["mean_ci_upper"].to_numpy(),
            alpha=0.15
        )

plt.axhline(0, color="black", linewidth=1)
plt.xlabel(x_label)
plt.ylabel("PC1 delta (Session 3 - Session 1)")
plt.title("PC1 change vs XTC dose, with separate slopes by dosage group")
plt.legend()
plt.tight_layout()
plt.savefig("pc1_delta_vs_dose_by_group.png", dpi=300, bbox_inches="tight")
print("Saved: pc1_delta_vs_dose_by_group.png")