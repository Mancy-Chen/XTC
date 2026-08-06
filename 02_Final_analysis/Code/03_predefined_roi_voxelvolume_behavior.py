"""Raw/adjusted ROI and BrainSegVol change associations with delayed recall."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import rankdata, shapiro, spearmanr

from config import BRAINSEGVOL_INPUT, CORR_ROI_VOLUME_OUT, GROUP_ORDER, N_PERMUTATIONS, RANDOM_SEED, ROI_VOLUME_COLUMNS, WHOLE_BRAIN_INPUT
from utils import add_group, apply_fdr, ensure_dirs, read_csv_numeric


def residualize(y: pd.Series, predictors: pd.DataFrame) -> pd.Series:
    frame = pd.concat([pd.to_numeric(y, errors="coerce").rename("y"), predictors.apply(pd.to_numeric, errors="coerce")], axis=1)
    complete = frame.dropna()
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    model = sm.OLS(complete["y"], sm.add_constant(complete.drop(columns="y"))).fit()
    out.loc[complete.index] = model.resid
    return out


def _rowwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx = rankdata(x, axis=1)
    ry = rankdata(y, axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    denominator = np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
    return (rx * ry).sum(axis=1) / denominator


def permute_group_difference(x: np.ndarray, y: np.ndarray, group: np.ndarray, n_perm: int, seed: int) -> dict:
    """Vectorized label permutation for the difference in subgroup Spearman rho."""
    rng = np.random.default_rng(seed)
    naive = group == "XTC-naive"
    users = group == "XTC users"
    rho_n = float(spearmanr(x[naive], y[naive]).statistic)
    rho_u = float(spearmanr(x[users], y[users]).statistic)
    observed = rho_n - rho_u
    n_naive = int(naive.sum())
    n_total = len(group)
    exceed = 0
    completed = 0
    batch_size = 1000
    while completed < n_perm:
        batch = min(batch_size, n_perm - completed)
        # Sorting random uniforms creates independent random permutations in compiled NumPy code.
        order = np.argsort(rng.random((batch, n_total)), axis=1)
        n_idx = order[:, :n_naive]
        u_idx = order[:, n_naive:]
        diff = _rowwise_spearman(x[n_idx], y[n_idx]) - _rowwise_spearman(x[u_idx], y[u_idx])
        exceed += int(np.count_nonzero(np.abs(diff) >= abs(observed)))
        completed += batch
    return {"rho_XTC_naive": rho_n, "rho_XTC_users": rho_u, "rho_difference_naive_minus_users": observed, "permutation_p": (exceed + 1) / (n_perm + 1), "n_permutations": n_perm}


def main() -> None:
    ensure_dirs([CORR_ROI_VOLUME_OUT])
    rad = read_csv_numeric(WHOLE_BRAIN_INPUT)
    brain = read_csv_numeric(BRAINSEGVOL_INPUT)
    data = add_group(rad.merge(brain[["subject_id", "aseg+DKT_BrainSegVol_delta"]], on="subject_id", how="inner", validate="one_to_one"))

    data["brainseg_pre_cm3"] = data["aseg+DKT_BrainSegVol_pre"]
    data["raw_BrainSegVol"] = data["aseg+DKT_BrainSegVol_delta"] / 1000.0
    data["adjusted_BrainSegVol"] = residualize(data["raw_BrainSegVol"], data[["brainseg_pre_cm3"]])

    outcome_columns = {"BrainSegVol": {"raw": "raw_BrainSegVol", "adjusted": "adjusted_BrainSegVol"}}
    for outcome, base in ROI_VOLUME_COLUMNS.items():
        raw_col = f"raw_{base}"
        adjusted_col = f"adjusted_{base}"
        data[raw_col] = pd.to_numeric(data[f"{base}_delta"], errors="coerce")
        data[adjusted_col] = residualize(data[raw_col], data[[f"{base}_pre", "brainseg_pre_cm3"]])
        outcome_columns[outcome] = {"raw": raw_col, "adjusted": adjusted_col}

    export_cols = ["subject_id", "xtc_group", "vwrec_pre", "vwrec_delta", "brainseg_pre_cm3"]
    for spec in outcome_columns.values(): export_cols.extend(spec.values())
    data[export_cols].to_csv(CORR_ROI_VOLUME_OUT / "roi_voxelvolume_raw_adjusted_change_dataset.csv", index=False)

    normality_rows = []
    for label, col in [("RAVLT delayed-recall change", "vwrec_delta")]:
        values = data[col].dropna()
        stat, p = shapiro(values)
        normality_rows.append({"variable": label, "column": col, "N": len(values), "Shapiro_W": stat, "p": p})
    for outcome, spec in outcome_columns.items():
        for adjustment, col in spec.items():
            values = data[col].dropna(); stat, p = shapiro(values)
            normality_rows.append({"variable": f"{adjustment} {outcome} change", "column": col, "N": len(values), "Shapiro_W": stat, "p": p})
    pd.DataFrame(normality_rows).to_csv(CORR_ROI_VOLUME_OUT / "normality_tests.csv", index=False)

    all_corr = []
    for sample_name, sample in [("Overall", data)] + [(g, data.loc[data.xtc_group.eq(g)]) for g in GROUP_ORDER]:
        for adjustment in ("raw", "adjusted"):
            family = []
            for outcome, spec in outcome_columns.items():
                subset = sample[[spec[adjustment], "vwrec_delta"]].dropna()
                rho, p = spearmanr(subset[spec[adjustment]], subset["vwrec_delta"])
                family.append({"sample": sample_name, "adjustment": adjustment, "outcome": outcome, "N": len(subset), "Spearman_rho": float(rho), "p": float(p)})
            all_corr.append(apply_fdr(pd.DataFrame(family)))
    correlations = pd.concat(all_corr, ignore_index=True)
    correlations.to_csv(CORR_ROI_VOLUME_OUT / "roi_voxelvolume_raw_adjusted_spearman.csv", index=False)
    correlations.query("sample != 'Overall' and adjustment == 'adjusted'").to_csv(CORR_ROI_VOLUME_OUT / "table_groupwise_adjusted_spearman.csv", index=False)

    # Supplementary comparison is pre-specified for the two highlighted adjusted outcomes.
    selected = {"BrainSegVol": outcome_columns["BrainSegVol"]["adjusted"], "Right hippocampus": outcome_columns["Right hippocampus"]["adjusted"]}
    comparison_rows = []
    for i, (outcome, col) in enumerate(selected.items()):
        subset = data[[col, "vwrec_delta", "xtc_group"]].dropna()
        result = permute_group_difference(subset[col].to_numpy(), subset["vwrec_delta"].to_numpy(), subset["xtc_group"].to_numpy(), N_PERMUTATIONS, RANDOM_SEED + i)
        result["outcome"] = outcome
        comparison_rows.append(result)
    comparisons = apply_fdr(pd.DataFrame(comparison_rows), p_col="permutation_p", output_col="FDR_q")
    comparisons.to_csv(CORR_ROI_VOLUME_OUT / "selected_adjusted_correlation_group_permutations.csv", index=False)

    slope_rows = []
    for outcome, col in selected.items():
        d = data[[col, "vwrec_delta", "xtc_group"]].dropna().copy()
        d["xtc_group"] = pd.Categorical(d["xtc_group"], categories=["XTC-naive", "XTC users"])
        model = smf.ols(f"vwrec_delta ~ {col} * C(xtc_group)", data=d).fit()
        interaction = next((t for t in model.params.index if ":" in t), None)
        slope_rows.append({"outcome": outcome, "interaction_term": interaction, "B": model.params.get(interaction, np.nan), "SE": model.bse.get(interaction, np.nan), "p": model.pvalues.get(interaction, np.nan), "N": int(model.nobs)})
    pd.DataFrame(slope_rows).to_csv(CORR_ROI_VOLUME_OUT / "selected_ols_slope_interactions.csv", index=False)
    print("ROI/BrainSegVol raw and adjusted Spearman analyses completed.")


if __name__ == "__main__":
    main()
