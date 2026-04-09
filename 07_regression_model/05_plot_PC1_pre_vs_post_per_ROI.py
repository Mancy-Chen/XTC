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

    order = ["sessie1", "sessie3"]
    data = [sub.loc[sub["session"] == s, "PC1"].dropna() for s in order]

    plt.figure(figsize=(5, 5), dpi=300)
    plt.boxplot(data, labels=order)
    plt.ylabel("PC1 score")
    plt.title(f"{roi}: Shape-PC1 pre vs post")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{roi}_PC1_pre_post_boxplot.png"), dpi=300)
    # plt.show()