"""Raw/adjusted ROI and BrainSegVol change associations with delayed recall."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro, spearmanr
import statsmodels.formula.api as smf

from config import (BEHAVIOR_INPUT, BRAINSEGVOL_INPUT, CORR_ROI_VOLUME_OUT, GROUP_ORDER,
                    N_ASSOCIATION_BOOTSTRAP, ASSOCIATION_BOOTSTRAP_SEED, FDR_ALPHA,
                    POLYSUBSTANCE_COLS, ROI_VOLUME_COLUMNS, WHOLE_BRAIN_INPUT)
from utils import add_group, apply_fdr, ensure_dirs, read_csv_numeric
from association_bootstrap import bootstrap_selected_correlations


def residualize(y: pd.Series, predictors: pd.DataFrame) -> pd.Series:
    frame = pd.concat([pd.to_numeric(y, errors="coerce").rename("y"), predictors.apply(pd.to_numeric, errors="coerce")], axis=1)
    complete = frame.dropna()
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    model = sm.OLS(complete["y"], sm.add_constant(complete.drop(columns="y"))).fit()
    out.loc[complete.index] = model.resid
    return out



def run_polysubstance_sensitivity(data: pd.DataFrame, outcome_columns: dict) -> None:
    """Pooled OLS sensitivity analysis of adjusted imaging change vs delayed-recall change.

    Models include imaging change, XTC group, their interaction, baseline delayed recall,
    behavioural-assessment age, sex, IQ, and ln(1 + x)-transformed cannabis, tobacco,
    alcohol, amphetamine and cocaine use. HC3 robust inference is used. Imaging-change
    predictors are standardized across the pooled sample. BH FDR is applied across the
    five imaging outcomes separately for each contrast.
    """
    behavior = read_csv_numeric(BEHAVIOR_INPUT).rename(columns={"studnr": "subject_id"})
    if "age" not in behavior.columns:
        raise KeyError("Behavioral input is missing age required for the N=95 sensitivity analysis")
    model_data = data.merge(
        behavior[["subject_id", "age"]].rename(columns={"age": "behavior_age"}),
        on="subject_id", how="left", validate="one_to_one",
    )
    required = ["vwrec_delta", "vwrec_pre", "behavior_age", "sex", "iq", *POLYSUBSTANCE_COLS.values()]
    missing = [col for col in required if col not in model_data.columns]
    if missing:
        raise KeyError(f"Sensitivity analysis is missing required columns: {missing}")
    if model_data[required].isna().any().any():
        counts = model_data[required].isna().sum(); counts = counts[counts.gt(0)].to_dict()
        raise ValueError(f"Sensitivity analysis requires complete N=95 covariates; missing: {counts}")

    rows = []
    for outcome, spec in outcome_columns.items():
        frame = model_data.copy()
        x = pd.to_numeric(frame[spec["adjusted"]], errors="coerce")
        frame["imaging_change_z"] = (x - x.mean()) / x.std(ddof=1)
        if frame["imaging_change_z"].isna().any():
            raise ValueError(f"{outcome}: adjusted imaging change contains missing values")
        model = smf.ols(
            "vwrec_delta ~ imaging_change_z * xtc_user + vwrec_pre + behavior_age + C(sex) + iq "
            "+ lca1jt + lsigpw + lalupw + ls1jht + lco1jt",
            data=frame,
        ).fit(cov_type="HC3")
        contrasts = [
            ("XTC-naive", "imaging_change_z"),
            ("XTC users", "imaging_change_z + imaging_change_z:xtc_user"),
            ("Group interaction", "imaging_change_z:xtc_user"),
        ]
        for contrast, hypothesis in contrasts:
            test = model.t_test(hypothesis)
            ci = np.asarray(test.conf_int()).reshape(-1, 2)[0]
            rows.append({
                "outcome": outcome,
                "contrast": contrast,
                "N": int(model.nobs),
                "beta_per_sd": float(np.asarray(test.effect).reshape(-1)[0]),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "p": float(np.asarray(test.pvalue).reshape(-1)[0]),
            })
    result = pd.DataFrame(rows)
    result["FDR_q"] = np.nan
    for contrast in result["contrast"].unique():
        mask = result["contrast"].eq(contrast)
        result.loc[mask, "FDR_q"] = apply_fdr(result.loc[mask], output_col="FDR_q")["FDR_q"].to_numpy()
    result.to_csv(CORR_ROI_VOLUME_OUT / "polysubstance_adjusted_imaging_memory_sensitivity.csv", index=False)


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

    run_polysubstance_sensitivity(data, outcome_columns)

    bootstrap_selected_correlations(
        data, correlations, ROI_VOLUME_COLUMNS, CORR_ROI_VOLUME_OUT,
        n_boot=N_ASSOCIATION_BOOTSTRAP, seed=ASSOCIATION_BOOTSTRAP_SEED,
        alpha=FDR_ALPHA,
    )
    print("ROI/BrainSegVol Spearman and polysubstance-adjusted sensitivity analyses completed.")


if __name__ == "__main__":
    main()
