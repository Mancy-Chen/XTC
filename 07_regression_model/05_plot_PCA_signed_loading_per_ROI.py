import os
import pandas as pd
import matplotlib.pyplot as plt

# input_dir = "/data/projects/CSC/code/XTC/07_regression_model/Output/shape_pc1_mixed_model"
input_dir = "/data/projects/CSC/code/XTC/07_regression_model/Output/shape_pc1_mixed_model_bilateral"
output_dir = os.path.join(input_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

roi_files = {
    # Left and right
    # "Left_Hippocampus": os.path.join(input_dir, "Left_Hippocampus_shape_PC1_loadings.csv"),
    # "Right_Hippocampus": os.path.join(input_dir, "Right_Hippocampus_shape_PC1_loadings.csv"),
    # "Left_Thalamus": os.path.join(input_dir, "Left_Thalamus_shape_PC1_loadings.csv"),
    # "Right_Thalamus": os.path.join(input_dir, "Right_Thalamus_shape_PC1_loadings.csv"),
    # Bilateral
    "Hippocampus": os.path.join(input_dir, "Hippocampus_shape_PC1_loadings.csv"),
    "Thalamus": os.path.join(input_dir, "Thalamus_shape_PC1_loadings.csv")
}

for roi, file_path in roi_files.items():
    df = pd.read_csv(file_path)

    # Keep only non-zero loadings
    plot_df = df[df["loading_PC1"] != 0].copy()

    # Optional: shorten labels for cleaner plotting
    plot_df["short_feature"] = (
        plot_df["feature"]
        .str.replace(f"{roi}_", "", regex=False)
        .str.replace("original_", "", regex=False)
        .str.replace("original-", "", regex=False)
    )

    # Sort by signed loading
    plot_df = plot_df.sort_values("loading_PC1", ascending=True)

    # muted blue for negative, muted coral/red for positive
    colors = ["#4C78A8" if x < 0 else "#E15759" for x in plot_df["loading_PC1"]]

    plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))), dpi=300)
    plt.barh(plot_df["short_feature"], plot_df["loading_PC1"], color=colors)
    plt.axvline(0, color="black", linewidth=1)

    plt.xlabel("Signed loading on PC1")
    plt.ylabel("Feature")
    plt.title(f"{roi}: Non-zero PC1 loadings")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{roi}_all_nonzero_signed_PC1_loadings.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # plt.show()

    print("Saved:", out_path)