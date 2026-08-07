"""Create manuscript/supplementary plots from the reorganized outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    COLORS,
    CORR_ROI_VOLUME_OUT,
    GROUP_ORDER,
    PCA_PREDEFINED_OUT,
    PCA_WHOLE_BRAIN_OUT,
    PLOT_PREDEFINED_PCA_OUT,
    PLOT_ROI_VOLUME_OUT,
    PLOT_WHOLE_BRAIN_PCA_OUT,
)
from utils import add_group, ensure_dirs, read_csv_numeric


def regression_line(ax, x: pd.Series, y: pd.Series, color: str) -> None:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2:
        return
    coefficients = np.polyfit(x[mask], y[mask], deg=1)
    grid = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(grid, coefficients[0] * grid + coefficients[1], color=color, linewidth=2)


def grouped_scatter(data: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for group in GROUP_ORDER:
        subset = data.loc[data["xtc_group"] == group]
        ax.scatter(
            subset[x], subset[y], label=f"{group} (n={len(subset)})",
            alpha=0.75, s=45, color=COLORS[group], edgecolors="none",
        )
        regression_line(ax, subset[x], subset[y], COLORS[group])
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def loading_plot(loadings: pd.DataFrame, label_col: str, loading_col: str, title: str, path, top_n: int = 20) -> None:
    plot_data = loadings.dropna(subset=[loading_col]).copy()
    plot_data["absolute_loading"] = plot_data[loading_col].abs()
    plot_data = plot_data.nlargest(top_n, "absolute_loading").sort_values(loading_col)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(plot_data[label_col], plot_data[loading_col])
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs([PLOT_ROI_VOLUME_OUT, PLOT_PREDEFINED_PCA_OUT, PLOT_WHOLE_BRAIN_PCA_OUT])

    roi = read_csv_numeric(CORR_ROI_VOLUME_OUT / "roi_voxelvolume_adjusted_change_dataset.csv")
    grouped_scatter(
        roi,
        "adjusted_right_hippocampus",
        "vwrec_delta",
        "Adjusted Δ right hippocampal volume (cm³)",
        "Δ delayed recall",
        "Adjusted right hippocampal change and delayed-recall change",
        PLOT_ROI_VOLUME_OUT / "adjusted_right_hippocampus_vs_delta_vwrec.png",
    )
    grouped_scatter(
        roi,
        "adjusted_BrainSegVol",
        "vwrec_delta",
        "Adjusted Δ BrainSegVol (cm³)",
        "Δ delayed recall",
        "Adjusted BrainSegVol change and delayed-recall change",
        PLOT_ROI_VOLUME_OUT / "adjusted_brainsegvol_vs_delta_vwrec.png",
    )

    pre_scores = add_group(read_csv_numeric(PCA_PREDEFINED_OUT / "predefined_roi_pca_scores_wide.csv"))
    pre_loadings = read_csv_numeric(PCA_PREDEFINED_OUT / "predefined_roi_pca_loadings.csv")
    for analysis, label in [
        ("Combined_shape_only", "Combined shape-only"),
        ("Combined_shape_firstorder", "Combined shape + first-order"),
    ]:
        subset_loadings = pre_loadings.loc[pre_loadings["analysis"] == analysis]
        loading_plot(
            subset_loadings,
            "feature",
            "PC1_loading",
            f"{label}: top PC1 loadings",
            PLOT_PREDEFINED_PCA_OUT / f"{analysis}_top20_PC1_loadings.png",
        )
        grouped_scatter(
            pre_scores,
            "log1p_xtc",
            f"{analysis}_PC1_delta",
            "ln(1 + cumulative XTC dose)",
            "Δ PC1",
            f"{label}: PC1 change versus XTC dose",
            PLOT_PREDEFINED_PCA_OUT / f"{analysis}_delta_PC1_vs_XTC_dose.png",
        )
        grouped_scatter(
            pre_scores,
            f"{analysis}_PC1_delta",
            "vwrec_delta",
            "Δ PC1",
            "Δ delayed recall",
            f"{label}: PC1 change and delayed-recall change",
            PLOT_PREDEFINED_PCA_OUT / f"{analysis}_delta_PC1_vs_delta_vwrec.png",
        )

    whole_scores = add_group(read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_scores_wide.csv"))
    whole_loadings = read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_loadings.csv")
    loading_plot(
        whole_loadings,
        "feature",
        "PC1_loading",
        "Whole-brain VoxelVolume PCA: top PC1 loadings",
        PLOT_WHOLE_BRAIN_PCA_OUT / "whole_brain_top20_PC1_loadings.png",
    )
    grouped_scatter(
        whole_scores,
        "log1p_xtc",
        "PC1_delta",
        "ln(1 + cumulative XTC dose)",
        "Δ whole-brain PC1",
        "Whole-brain PC1 change versus XTC dose",
        PLOT_WHOLE_BRAIN_PCA_OUT / "whole_brain_delta_PC1_vs_XTC_dose.png",
    )
    grouped_scatter(
        whole_scores,
        "PC1_delta",
        "vwrec_delta",
        "Δ whole-brain PC1",
        "Δ delayed recall",
        "Whole-brain PC1 change and delayed-recall change",
        PLOT_WHOLE_BRAIN_PCA_OUT / "whole_brain_delta_PC1_vs_delta_vwrec.png",
    )
    print("Plots completed.")


if __name__ == "__main__":
    main()
