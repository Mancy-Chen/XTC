"""Raw/adjusted ROI and BrainSegVol change associations with delayed recall."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro, spearmanr

from config import BRAINSEGVOL_INPUT, CORR_ROI_VOLUME_OUT, GROUP_ORDER, N_ASSOCIATION_BOOTSTRAP, ASSOCIATION_BOOTSTRAP_SEED, FDR_ALPHA, ROI_VOLUME_COLUMNS, WHOLE_BRAIN_INPUT
from utils import add_group, apply_fdr, ensure_dirs, read_csv_numeric
from association_bootstrap import bootstrap_selected_correlations


def residualize(y: pd.Series, predictors: pd.DataFrame) -> pd.Series:
    frame = pd.concat([pd.to_numeric(y, errors="coerce").rename("y"), predictors.apply(pd.to_numeric, errors="coerce")], axis=1)
    complete = frame.dropna()
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    model = sm.OLS(complete["y"], sm.add_constant(complete.drop(columns="y"))).fit()
    out.loc[complete.index] = model.resid
    return out


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

    bootstrap_selected_correlations(
        data, correlations, ROI_VOLUME_COLUMNS, CORR_ROI_VOLUME_OUT,
        n_boot=N_ASSOCIATION_BOOTSTRAP, seed=ASSOCIATION_BOOTSTRAP_SEED,
        alpha=FDR_ALPHA,
    )
    print("ROI/BrainSegVol raw and adjusted Spearman analyses completed.")


if __name__ == "__main__":
    main()
