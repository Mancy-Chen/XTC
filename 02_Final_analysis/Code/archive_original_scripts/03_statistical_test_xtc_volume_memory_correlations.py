#!/usr/bin/env python3
"""
Residualize longitudinal BrainSegVol and right-hippocampal changes,
calculate group-specific Spearman correlations with RAVLT delayed-recall
change, compare the correlations by permutation, test the plotted OLS
slope interaction, apply FDR correction, and create both final plots.

Install:
    pip install pandas numpy scipy statsmodels matplotlib

Run in PyCharm:
    1. Open this file.
    2. Right-click in the editor.
    3. Choose "Run File in Python Console".

Or, after executing the file definitions in the Python Console:
    main()

The paths and settings are defined near the top of this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


COLORS = {
    "XTC users": "#eb5600",
    "XTC-naive": "#1a9988",
}
GROUP_ORDER = ["XTC users", "XTC-naive"]

OUTCOMES = {
    "BrainSegVol": {
        "adjusted_col": "adjusted_delta_brainsegvol",
        "title": "Adjusted BrainSegVol change and delayed-recall change",
        "xlabel": (
            "Adjusted Δ BrainSegVol (cm³)\n"
            "Residualized for baseline BrainSegVol"
        ),
        "filename": (
            "adjusted_delta_brainsegvol_"
            "users_orange_naive_green.png"
        ),
    },
    "Right hippocampus": {
        "adjusted_col": "adjusted_delta_right_hippocampus",
        "title": (
            "Adjusted right hippocampal volume change and "
            "delayed-recall change"
        ),
        "xlabel": (
            "Adjusted Δ right hippocampal volume (cm³)\n"
            "Residualized for baseline right hippocampal volume "
            "and baseline BrainSegVol"
        ),
        "filename": (
            "adjusted_delta_right_hippocampus_"
            "users_orange_naive_green.png"
        ),
    },
}


# =========================================================
# PROJECT PATHS AND ANALYSIS SETTINGS
# Edit these values directly when needed.
# This version does not use command-line arguments, so it works in the
# PyCharm Python Console and with "Run File in Python Console".
# =========================================================
RADIOMICS_DATA = Path(
    "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/"
    "02_filteredQC/New/"
    "radiomics_voxelvolume_pre_delta_95_subjects_cm3_with_xtc_ravlt_covariates.csv"
)

BRAINSEG_DATA = Path(
    "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/"
    "02_filteredQC/whole_brain_volume/"
    "whole_brain_volume_pre_delta.csv"
)

OUTPUT_DIR = Path(
    "/data/projects/CSC/code/XTC/07_regression_model/Output/"
    "Radiomics_statistics"
)

N_PERMUTATIONS = 100000
N_BOOTSTRAP = 10000
RANDOM_SEED = 20260728


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def load_data(
    radiomics_path: Path,
    brainseg_path: Path,
) -> pd.DataFrame:
    if not radiomics_path.exists():
        raise FileNotFoundError(radiomics_path)
    if not brainseg_path.exists():
        raise FileNotFoundError(brainseg_path)

    rad = pd.read_csv(radiomics_path, low_memory=False)
    brain = pd.read_csv(brainseg_path, low_memory=False)

    require_columns(
        rad,
        [
            "subject_id",
            "xlttot_sessie3",
            "vwrec_delta",
            "right_hippocampus_pre",
            "right_hippocampus_delta",
        ],
        "Radiomics file",
    )
    require_columns(
        brain,
        [
            "subject_id",
            "aseg+DKT_BrainSegVol_pre",
            "aseg+DKT_BrainSegVol_delta",
        ],
        "BrainSegVol file",
    )

    rad = rad.copy()
    brain = brain.copy()
    for frame in (rad, brain):
        frame["subject_id"] = (
            frame["subject_id"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if rad["subject_id"].duplicated().any():
        raise ValueError("Duplicate subject IDs in radiomics file.")
    if brain["subject_id"].duplicated().any():
        raise ValueError("Duplicate subject IDs in BrainSegVol file.")

    data = rad[
        [
            "subject_id",
            "xlttot_sessie3",
            "vwrec_delta",
            "right_hippocampus_pre",
            "right_hippocampus_delta",
        ]
    ].merge(
        brain[
            [
                "subject_id",
                "aseg+DKT_BrainSegVol_pre",
                "aseg+DKT_BrainSegVol_delta",
            ]
        ],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )

    numeric_columns = [
        "xlttot_sessie3",
        "vwrec_delta",
        "right_hippocampus_pre",
        "right_hippocampus_delta",
        "aseg+DKT_BrainSegVol_pre",
        "aseg+DKT_BrainSegVol_delta",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=numeric_columns).copy()

    data["group"] = np.where(
        data["xlttot_sessie3"] > 0,
        "XTC users",
        "XTC-naive",
    )
    data["delta_vwrec"] = data["vwrec_delta"]

    # Dedicated whole-brain file is in mm³.
    data["brainseg_pre_cm3"] = (
        data["aseg+DKT_BrainSegVol_pre"] / 1000.0
    )
    data["brainseg_delta_cm3"] = (
        data["aseg+DKT_BrainSegVol_delta"] / 1000.0
    )

    # Radiomics voxel-volume columns are already in cm³.
    data["right_hippocampus_pre_cm3"] = (
        data["right_hippocampus_pre"]
    )
    data["right_hippocampus_delta_cm3"] = (
        data["right_hippocampus_delta"]
    )
    return data


def residualize(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    brainseg_model = sm.OLS(
        result["brainseg_delta_cm3"],
        sm.add_constant(result[["brainseg_pre_cm3"]]),
    ).fit()
    result["adjusted_delta_brainsegvol"] = brainseg_model.resid

    hippo_model = sm.OLS(
        result["right_hippocampus_delta_cm3"],
        sm.add_constant(
            result[
                [
                    "right_hippocampus_pre_cm3",
                    "brainseg_pre_cm3",
                ]
            ]
        ),
    ).fit()
    result["adjusted_delta_right_hippocampus"] = (
        hippo_model.resid
    )

    return result


def groupwise_spearman(
    data: pd.DataFrame,
    outcome: str,
    adjusted_col: str,
) -> pd.DataFrame:
    rows = []
    for group_name in GROUP_ORDER:
        sub = data.loc[
            data["group"] == group_name,
            [adjusted_col, "delta_vwrec"],
        ].dropna()
        rho, p_value = spearmanr(
            sub[adjusted_col],
            sub["delta_vwrec"],
        )
        rows.append(
            {
                "Outcome": outcome,
                "Group": group_name,
                "N": len(sub),
                "Spearman_rho": float(rho),
                "p": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def rho_difference(
    x: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
) -> tuple[float, float, float]:
    users = group == "XTC users"
    naive = group == "XTC-naive"

    rho_users = float(spearmanr(x[users], y[users]).statistic)
    rho_naive = float(spearmanr(x[naive], y[naive]).statistic)

    return rho_users, rho_naive, rho_naive - rho_users


def permutation_compare_correlations(
    x: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    rho_users, rho_naive, observed = rho_difference(x, y, group)

    permutation_differences = np.empty(n_permutations)
    for index in range(n_permutations):
        shuffled_group = rng.permutation(group)
        permutation_differences[index] = rho_difference(
            x,
            y,
            shuffled_group,
        )[2]

    p_value = (
        np.sum(np.abs(permutation_differences) >= abs(observed))
        + 1
    ) / (n_permutations + 1)

    return rho_users, rho_naive, observed, float(p_value)


def bootstrap_rho_difference_ci(
    x: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    users_indices = np.flatnonzero(group == "XTC users")
    naive_indices = np.flatnonzero(group == "XTC-naive")

    differences = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        users_sample = rng.choice(
            users_indices,
            size=len(users_indices),
            replace=True,
        )
        naive_sample = rng.choice(
            naive_indices,
            size=len(naive_indices),
            replace=True,
        )

        rho_users = spearmanr(
            x[users_sample],
            y[users_sample],
        ).statistic
        rho_naive = spearmanr(
            x[naive_sample],
            y[naive_sample],
        ).statistic
        differences[index] = rho_naive - rho_users

    lower, upper = np.percentile(differences, [2.5, 97.5])
    return float(lower), float(upper)


def slope_interaction(
    data: pd.DataFrame,
    adjusted_col: str,
) -> dict[str, float]:
    model_data = data[
        [adjusted_col, "delta_vwrec", "group"]
    ].dropna().copy()

    model_data["group"] = pd.Categorical(
        model_data["group"],
        categories=["XTC users", "XTC-naive"],
    )

    model = smf.ols(
        f"delta_vwrec ~ {adjusted_col} * C(group)",
        data=model_data,
    ).fit()

    interaction_terms = [
        term
        for term in model.params.index
        if ":" in term and adjusted_col in term
    ]
    if len(interaction_terms) != 1:
        raise RuntimeError(
            "Could not identify the group-by-volume interaction."
        )

    term = interaction_terms[0]
    ci = model.conf_int().loc[term]

    return {
        "OLS_interaction_estimate": float(model.params[term]),
        "OLS_interaction_CI_lower": float(ci.iloc[0]),
        "OLS_interaction_CI_upper": float(ci.iloc[1]),
        "OLS_interaction_p": float(model.pvalues[term]),
    }


def add_group_plot(
    ax: plt.Axes,
    sub: pd.DataFrame,
    adjusted_col: str,
    group_name: str,
) -> None:
    color = COLORS[group_name]

    ax.scatter(
        sub[adjusted_col],
        sub["delta_vwrec"],
        s=52,
        alpha=0.82,
        color=color,
        edgecolors="white",
        linewidths=0.6,
        label=group_name,
        zorder=3,
    )

    x = sub[adjusted_col].to_numpy()
    y = sub["delta_vwrec"].to_numpy()

    # Visual linear fit; the formal nonparametric test is Spearman.
    model = sm.OLS(y, sm.add_constant(x)).fit()
    x_grid = np.linspace(x.min(), x.max(), 250)
    prediction = model.get_prediction(
        sm.add_constant(x_grid)
    ).summary_frame(alpha=0.05)

    ax.plot(
        x_grid,
        prediction["mean"].to_numpy(),
        color=color,
        linewidth=2.5,
        zorder=4,
    )
    ax.fill_between(
        x_grid,
        prediction["mean_ci_lower"].to_numpy(),
        prediction["mean_ci_upper"].to_numpy(),
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )


def make_plot(
    data: pd.DataFrame,
    adjusted_col: str,
    title: str,
    xlabel: str,
    group_results: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.7))

    for group_name in GROUP_ORDER:
        sub = data.loc[
            data["group"] == group_name,
            [adjusted_col, "delta_vwrec"],
        ].dropna()
        add_group_plot(ax, sub, adjusted_col, group_name)

    ax.axhline(0, color="#b8b8b8", linewidth=1.0, zorder=0)
    ax.axvline(0, color="#b8b8b8", linewidth=1.0, zorder=0)

    ax.set_title(title, fontsize=18, pad=14, weight="semibold")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Δ RAVLT delayed recall (vwrec)", fontsize=13)

    lookup = group_results.set_index("Group").to_dict("index")
    annotation = (
        f"XTC users: N={lookup['XTC users']['N']}, "
        f"ρ={lookup['XTC users']['Spearman_rho']:.3f}, "
        f"p={lookup['XTC users']['p']:.3f}\n"
        f"XTC-naive: N={lookup['XTC-naive']['N']}, "
        f"ρ={lookup['XTC-naive']['Spearman_rho']:.3f}, "
        f"p={lookup['XTC-naive']['p']:.3f}"
    )

    ax.text(
        0.025,
        0.965,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11.5,
        bbox={
            "facecolor": "white",
            "alpha": 0.92,
            "edgecolor": "#666666",
            "boxstyle": "round,pad=0.35",
        },
    )

    ax.legend(
        title="Group",
        loc="upper right",
        frameon=True,
        fontsize=11,
        title_fontsize=11,
    )
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.012,
        (
            "Spearman correlations are reported; lines and shaded "
            "bands show visual OLS fits with 95% confidence intervals."
        ),
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if N_PERMUTATIONS < 1:
        raise ValueError("N_PERMUTATIONS must be at least 1.")
    if N_BOOTSTRAP < 0:
        raise ValueError("N_BOOTSTRAP cannot be negative.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Radiomics input:")
    print(RADIOMICS_DATA)
    print("\nBrainSegVol input:")
    print(BRAINSEG_DATA)
    print("\nOutput directory:")
    print(OUTPUT_DIR)

    data = residualize(
        load_data(RADIOMICS_DATA, BRAINSEG_DATA)
    )

    permutation_rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_rng = np.random.default_rng(RANDOM_SEED + 1)

    groupwise_tables = []
    comparison_rows = []

    for outcome, spec in OUTCOMES.items():
        adjusted_col = spec["adjusted_col"]

        group_results = groupwise_spearman(
            data,
            outcome,
            adjusted_col,
        )
        groupwise_tables.append(group_results)

        complete = data[
            [adjusted_col, "delta_vwrec", "group"]
        ].dropna()

        x = complete[adjusted_col].to_numpy()
        y = complete["delta_vwrec"].to_numpy()
        group = complete["group"].to_numpy()

        (
            rho_users,
            rho_naive,
            difference,
            permutation_p,
        ) = permutation_compare_correlations(
            x,
            y,
            group,
            N_PERMUTATIONS,
            permutation_rng,
        )

        if N_BOOTSTRAP > 0:
            ci_lower, ci_upper = bootstrap_rho_difference_ci(
                x,
                y,
                group,
                N_BOOTSTRAP,
                bootstrap_rng,
            )
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        comparison_rows.append(
            {
                "Outcome": outcome,
                "N_total": len(complete),
                "N_XTC_users": int(
                    (complete["group"] == "XTC users").sum()
                ),
                "N_XTC_naive": int(
                    (complete["group"] == "XTC-naive").sum()
                ),
                "rho_XTC_users": rho_users,
                "rho_XTC_naive": rho_naive,
                "rho_difference_naive_minus_users": difference,
                "rho_difference_bootstrap_CI_lower": ci_lower,
                "rho_difference_bootstrap_CI_upper": ci_upper,
                "permutation_p": permutation_p,
                **slope_interaction(data, adjusted_col),
            }
        )

        make_plot(
            data=data,
            adjusted_col=adjusted_col,
            title=spec["title"],
            xlabel=spec["xlabel"],
            group_results=group_results,
            output_path=OUTPUT_DIR / spec["filename"],
        )

    groupwise_results = pd.concat(
        groupwise_tables,
        ignore_index=True,
    )
    comparison_results = pd.DataFrame(comparison_rows)

    comparison_results["permutation_FDR_q"] = multipletests(
        comparison_results["permutation_p"],
        method="fdr_bh",
    )[1]
    comparison_results["OLS_interaction_FDR_q"] = multipletests(
        comparison_results["OLS_interaction_p"],
        method="fdr_bh",
    )[1]

    data.to_csv(
        OUTPUT_DIR / "analysis_dataset_with_residuals.csv",
        index=False,
    )
    groupwise_results.to_csv(
        OUTPUT_DIR / "groupwise_spearman_results.csv",
        index=False,
    )
    comparison_results.to_csv(
        OUTPUT_DIR / "between_group_comparison_results.csv",
        index=False,
    )

    summary = [
        "Group-specific Spearman correlations",
        "=" * 70,
        groupwise_results.to_string(index=False),
        "",
        "Formal between-group comparisons",
        "=" * 70,
        comparison_results.to_string(index=False),
        "",
        "Notes",
        "=" * 70,
        (
            "permutation_p compares rho_XTC-naive with rho_XTC-users."
        ),
        (
            "OLS_interaction_p compares the ordinary linear slopes "
            "drawn on the plots; it is not a Spearman test."
        ),
        (
            "The shaded areas are 95% confidence intervals for the "
            "estimated mean OLS lines, not for Spearman rho."
        ),
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("\nGroup-specific Spearman correlations:")
    print(groupwise_results.to_string(index=False))
    print("\nFormal between-group comparisons:")
    print(comparison_results.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
