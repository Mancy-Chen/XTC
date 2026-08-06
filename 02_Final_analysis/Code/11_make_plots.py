"""Create manuscript and supplementary plots, including PC1-PC5 loadings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr

from config import (
    BOOTSTRAP_WHOLE_BRAIN_OUT, COLORS, CORR_PREDEFINED_PCA_OUT, CORR_ROI_VOLUME_OUT,
    CORR_WHOLE_BRAIN_PCA_OUT, GROUP_ORDER, PCA_PREDEFINED_OUT, PCA_WHOLE_BRAIN_OUT,
    PLOT_BOOTSTRAP_OUT, PLOT_PREDEFINED_PCA_OUT, PLOT_ROI_VOLUME_OUT, PLOT_WHOLE_BRAIN_PCA_OUT,
)
from utils import add_group, ensure_dirs, read_csv_numeric


# Keep points, fitted lines, and confidence bands aligned by group.
COLORS.update({
    "XTC users": "#eb5600",
    "XTC-naive": "#1a9988",
})


def format_p_value(p_value: float) -> str:
    """Format p values for plot annotations."""
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def regression_line_with_ci(
    ax,
    x: pd.Series,
    y: pd.Series,
    color: str,
    alpha: float = 0.18,
) -> None:
    """Draw a descriptive OLS line and its 95% confidence band."""
    mask = x.notna() & y.notna()
    x_clean = pd.to_numeric(x[mask], errors="coerce")
    y_clean = pd.to_numeric(y[mask], errors="coerce")
    valid = x_clean.notna() & y_clean.notna()
    x_clean = x_clean[valid].astype(float)
    y_clean = y_clean[valid].astype(float)

    if len(x_clean) < 3 or x_clean.nunique() < 2:
        return

    design = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, design).fit()

    grid = np.linspace(x_clean.min(), x_clean.max(), 200)
    prediction_design = sm.add_constant(grid)
    prediction = model.get_prediction(prediction_design).summary_frame(alpha=0.05)

    ax.plot(
        grid,
        prediction["mean"].to_numpy(),
        color=color,
        linewidth=2.8,
        zorder=4,
    )
    ax.fill_between(
        grid,
        prediction["mean_ci_lower"].to_numpy(),
        prediction["mean_ci_upper"].to_numpy(),
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=2,
    )


def grouped_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path,
    *,
    show_spearman: bool = False,
    add_vertical_zero: bool = False,
    figure_size: tuple[float, float] = (8, 6),
    footnote: str | None = None,
) -> None:
    """Create a group-stratified scatter plot with OLS 95% CIs."""
    fig, ax = plt.subplots(figsize=figure_size)
    correlation_results: dict[str, tuple[int, float, float]] = {}

    for group in GROUP_ORDER:
        subset = data.loc[data["xtc_group"] == group, [x, y]].dropna().copy()
        color = COLORS[group]

        ax.scatter(
            subset[x],
            subset[y],
            label=group,
            alpha=0.85,
            s=50,
            color=color,
            edgecolors="none",
            zorder=3,
        )
        regression_line_with_ci(ax, subset[x], subset[y], color)

        if show_spearman and len(subset) >= 3:
            rho, p_value = spearmanr(subset[x], subset[y])
            correlation_results[group] = (len(subset), float(rho), float(p_value))

    ax.axhline(0, linewidth=1, linestyle="--", color="#888888", alpha=0.8, zorder=1)
    if add_vertical_zero:
        ax.axvline(0, linewidth=1, linestyle="--", color="#888888", alpha=0.8, zorder=1)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=20, pad=12)
    ax.legend(title="Group", frameon=True, fontsize=13, title_fontsize=13)
    ax.tick_params(labelsize=13)

    if show_spearman:
        annotation_lines = []
        for group in ["XTC users", "XTC-naive"]:
            if group not in correlation_results:
                continue
            n, rho, p_value = correlation_results[group]
            annotation_lines.append(
                f"{group}: N={n}, ρ={rho:.3f}, p={format_p_value(p_value)}"
            )

        ax.text(
            0.02,
            0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.80,
            },
        )

    if footnote:
        fig.text(0.5, 0.02, footnote, ha="center", fontsize=11)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        fig.tight_layout()

    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def loading_plot(loadings: pd.DataFrame, label_col: str, loading_col: str, title: str, path, top_n: int = 20) -> None:
    plot_data = loadings.dropna(subset=[loading_col]).copy()
    plot_data["absolute_loading"] = plot_data[loading_col].abs()
    plot_data = plot_data.nlargest(top_n, "absolute_loading").sort_values(loading_col)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(plot_data[label_col], plot_data[loading_col])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(f"{loading_col.replace('_loading','')} loading"); ax.set_ylabel("Feature"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def scree_plot(explained: pd.DataFrame, title: str, path) -> None:
    d = explained.head(10).copy()
    x = np.arange(1, len(d) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, d["explained_variance_ratio"] * 100, marker="o")
    ax.set_xticks(x); ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def bootstrap_loading_plot(summary: pd.DataFrame, path) -> None:
    d = summary.nsmallest(20, "original_absolute_rank").sort_values("PC1_loading")
    lower = d["PC1_loading"] - d["percentile_2_5"]
    upper = d["percentile_97_5"] - d["PC1_loading"]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(d["PC1_loading"], d["feature"], xerr=np.vstack([lower, upper]), fmt="o", capsize=3)
    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_xlabel("Original PC1 loading with bootstrap percentile interval")
    ax.set_ylabel("Region"); ax.set_title("Bootstrap stability of top whole-brain PC1 loadings")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ensure_dirs([PLOT_ROI_VOLUME_OUT, PLOT_PREDEFINED_PCA_OUT, PLOT_WHOLE_BRAIN_PCA_OUT, PLOT_BOOTSTRAP_OUT])

    # Main Figure 1: corrected I221 data, adjusted right hippocampus only.
    # BrainSegVol is intentionally not plotted because its corrected
    # association with delayed-recall change is not statistically significant.
    roi = read_csv_numeric(CORR_ROI_VOLUME_OUT / "roi_voxelvolume_raw_adjusted_change_dataset.csv")
    grouped_scatter(
        roi,
        "adjusted_right_hippocampus",
        "vwrec_delta",
        "Adjusted Δ right hippocampal volume (cm³)\n"
        "Residualized for baseline right hippocampal volume and baseline BrainSegVol",
        "Δ RAVLT delayed recall (vwrec)",
        "Adjusted right hippocampal volume change and delayed-recall change",
        PLOT_ROI_VOLUME_OUT / "right_hippocampus_adjusted_vs_delta_vwrec.png",
        show_spearman=True,
        add_vertical_zero=True,
        figure_size=(12, 8),
        footnote=(
            "Spearman correlations are reported; lines and shaded bands show "
            "descriptive OLS fits with 95% confidence intervals."
        ),
    )

    pre_scores = add_group(read_csv_numeric(PCA_PREDEFINED_OUT / "predefined_roi_pca_scores_wide.csv"))
    pre_corr = read_csv_numeric(CORR_PREDEFINED_PCA_OUT / "predefined_roi_pca_raw_adjusted_dataset.csv")
    pre_scores = pre_scores.merge(pre_corr[[c for c in pre_corr.columns if c.endswith("_adjusted") or c == "subject_id"]], on="subject_id", how="left", validate="one_to_one")
    pre_loadings = read_csv_numeric(PCA_PREDEFINED_OUT / "predefined_roi_pca_loadings.csv")
    pre_explained = read_csv_numeric(PCA_PREDEFINED_OUT / "predefined_roi_pca_explained_variance.csv")
    for analysis, label in [("Combined_shape_only", "Combined shape-only"), ("Combined_shape_firstorder", "Combined shape + first-order")]:
        subset_loadings = pre_loadings.loc[pre_loadings["analysis"] == analysis]
        for pc in range(1, 6):
            loading_plot(subset_loadings, "feature", f"PC{pc}_loading", f"{label}: top PC{pc} loadings", PLOT_PREDEFINED_PCA_OUT / f"{analysis}_top20_PC{pc}_loadings.png")
        scree_plot(pre_explained.loc[pre_explained.analysis.eq(analysis)], f"{label}: explained variance", PLOT_PREDEFINED_PCA_OUT / f"{analysis}_scree_plot.png")
        grouped_scatter(pre_scores, f"{analysis}_PC1_delta", "vwrec_delta", "Raw ΔPC1", "Δ delayed recall", f"{label}: raw PC1 change and delayed recall", PLOT_PREDEFINED_PCA_OUT / f"{analysis}_raw_delta_PC1_vs_delta_vwrec.png")
        grouped_scatter(pre_scores, f"{analysis}_PC1_delta_adjusted", "vwrec_delta", "Adjusted ΔPC1", "Δ delayed recall", f"{label}: adjusted PC1 change and delayed recall", PLOT_PREDEFINED_PCA_OUT / f"{analysis}_adjusted_delta_PC1_vs_delta_vwrec.png")
        grouped_scatter(pre_scores, "log1p_xtc", f"{analysis}_PC1_delta", "ln(1 + cumulative XTC dose)", "ΔPC1", f"{label}: PC1 change versus XTC dose", PLOT_PREDEFINED_PCA_OUT / f"{analysis}_delta_PC1_vs_XTC_dose.png")

    whole_scores = add_group(read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_scores_wide.csv"))
    whole_corr = read_csv_numeric(CORR_WHOLE_BRAIN_PCA_OUT / "whole_brain_pc1_raw_adjusted_dataset.csv")
    whole_scores = whole_scores.merge(whole_corr[["subject_id", "PC1_delta_adjusted"]], on="subject_id", how="left", validate="one_to_one")
    whole_loadings = read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_loadings.csv")
    for pc in range(1, 6):
        loading_plot(whole_loadings, "feature", f"PC{pc}_loading", f"Whole-brain VoxelVolume PCA: top PC{pc} loadings", PLOT_WHOLE_BRAIN_PCA_OUT / f"whole_brain_top20_PC{pc}_loadings.png")
    scree_plot(read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_explained_variance.csv"), "Whole-brain VoxelVolume PCA: explained variance", PLOT_WHOLE_BRAIN_PCA_OUT / "whole_brain_scree_plot.png")
    grouped_scatter(whole_scores, "PC1_delta", "vwrec_delta", "Raw Δ whole-brain PC1", "Δ delayed recall", "Whole-brain raw PC1 change and delayed recall", PLOT_WHOLE_BRAIN_PCA_OUT / "whole_brain_raw_delta_PC1_vs_delta_vwrec.png")
    grouped_scatter(whole_scores, "PC1_delta_adjusted", "vwrec_delta", "Adjusted Δ whole-brain PC1", "Δ delayed recall", "Whole-brain adjusted PC1 change and delayed recall", PLOT_WHOLE_BRAIN_PCA_OUT / "whole_brain_adjusted_delta_PC1_vs_delta_vwrec.png")
    grouped_scatter(whole_scores, "log1p_xtc", "PC1_delta", "ln(1 + cumulative XTC dose)", "Δ whole-brain PC1", "Whole-brain PC1 change versus XTC dose", PLOT_WHOLE_BRAIN_PCA_OUT / "whole_brain_delta_PC1_vs_XTC_dose.png")

    bootstrap_loading_plot(read_csv_numeric(BOOTSTRAP_WHOLE_BRAIN_OUT / "whole_brain_pc1_loading_bootstrap_summary.csv"), PLOT_BOOTSTRAP_OUT / "whole_brain_PC1_top20_loading_bootstrap_intervals.png")
    print("All plots, including PC1-PC5 loading plots, completed.")


if __name__ == "__main__":
    main()
