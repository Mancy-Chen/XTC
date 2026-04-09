import matplotlib
matplotlib.use("Agg")   # safe backend

import pandas as pd
import matplotlib.pyplot as plt

# loadings_path = "/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_volume_pca_lmm_mean_age_imputed/whole_brain_pca_loadings.csv"
loadings_path = '/data/projects/CSC/code/XTC/07_regression_model/Output/whole_brain_pca_lmm_with_xtc_dose/whole_brain_pca_loadings.csv'
df = pd.read_csv(loadings_path)

pc_col = "PC1"
top_n = 20

plot_df = df[["feature", pc_col]].copy()
plot_df["abs_loading"] = plot_df[pc_col].abs()
plot_df = plot_df.sort_values("abs_loading", ascending=False).head(top_n)
plot_df = plot_df.sort_values(pc_col, ascending=True)

plot_df["feature_short"] = (
    plot_df["feature"]
    .str.replace("aseg+DKT_", "", regex=False)
    .str.replace("cerebellum.CerebNet_", "", regex=False)
    .str.replace("hypothalamus.HypVINN_", "", regex=False)
)

plt.figure(figsize=(10, 8))
plt.barh(plot_df["feature_short"], plot_df[pc_col])
plt.axvline(0, color="black", linewidth=1)
plt.xlabel(f"{pc_col} loading")
plt.ylabel("Feature")
plt.title(f"Top {top_n} loadings for {pc_col}")
plt.tight_layout()
plt.savefig("pc1_loadings_plot.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: pc1_loadings_plot.png")