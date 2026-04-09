import os
import pandas as pd
import matplotlib.pyplot as plt

input_dir = "/data/projects/CSC/code/XTC/07_regression_model/Output/shape_pc1_mixed_model"
output_dir = os.path.join(input_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

pc1_long_path = os.path.join(input_dir, "shape_pc1_long.csv")
pc1_df = pd.read_csv(pc1_long_path)

rois = pc1_df["roi"].unique()

for roi in rois:
    sub = pc1_df[pc1_df["roi"] == roi].copy()

    plt.figure(figsize=(6, 5), dpi=300)
    plt.hist(sub.loc[sub["session"] == "sessie1", "PC1"].dropna(), bins=20, alpha=0.6, label="sessie1")
    plt.hist(sub.loc[sub["session"] == "sessie3", "PC1"].dropna(), bins=20, alpha=0.6, label="sessie3")
    plt.xlabel("PC1 score")
    plt.ylabel("Count")
    plt.title(f"{roi}: PC1 distribution by session")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{roi}_PC1_histogram.png"), dpi=300)
    # plt.show()