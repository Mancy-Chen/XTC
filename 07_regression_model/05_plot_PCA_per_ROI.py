import os
import pandas as pd
import matplotlib.pyplot as plt

input_dir = "/data/projects/CSC/code/XTC/07_regression_model/Output/shape_pc1_mixed_model"
output_dir = os.path.join(input_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

roi_files = {
    "Left_Hippocampus": os.path.join(input_dir, "Left_Hippocampus_shape_PC1_loadings.csv"),
    "Right_Hippocampus": os.path.join(input_dir, "Right_Hippocampus_shape_PC1_loadings.csv"),
    "Left_Thalamus": os.path.join(input_dir, "Left_Thalamus_shape_PC1_loadings.csv"),
    "Right_Thalamus": os.path.join(input_dir, "Right_Thalamus_shape_PC1_loadings.csv"),
}

top_n = 10

for roi, file_path in roi_files.items():
    df = pd.read_csv(file_path)

    # top absolute loadings
    top_df = df.sort_values("abs_loading_PC1", ascending=False).head(top_n).copy()
    top_df = top_df.iloc[::-1]  # reverse for horizontal bar plot

    plt.figure(figsize=(10, 5), dpi=300)
    plt.barh(top_df["feature"], top_df["abs_loading_PC1"])
    plt.xlabel("Absolute loading on PC1")
    plt.ylabel("Feature")
    plt.title(f"{roi}: Top {top_n} contributors to PC1")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{roi}_top_abs_PC1_loadings.png"), dpi=300)
    # plt.show()