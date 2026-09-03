"""Raw/BrainSegVol-adjusted Spearman and OLS analyses for predefined ROI PCA."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from config import CORR_PREDEFINED_PCA_OUT, GROUP_ORDER, OLS_PREDEFINED_PCA_OUT, PCA_PREDEFINED_OUT, POLYSUBSTANCE_COLS
from utils import add_group, apply_fdr, centered, ensure_dirs, mean_impute, read_csv_numeric, tidy_model_result, write_text


def analysis_names(scores: pd.DataFrame) -> list[str]:
    suffix = "_PC1_pre"
    return [c[:-len(suffix)] for c in scores.columns if c.endswith(suffix)]


def residualize_delta(data: pd.DataFrame, analysis: str) -> pd.Series:
    cols = [f"{analysis}_PC1_delta", f"{analysis}_PC1_pre", "aseg+DKT_BrainSegVol_pre"]
    d = data[cols].dropna()
    model = sm.OLS(d[cols[0]], sm.add_constant(d[cols[1:]])).fit()
    out = pd.Series(np.nan, index=data.index, dtype=float)
    out.loc[d.index] = model.resid
    return out


def main() -> None:
    ensure_dirs([CORR_PREDEFINED_PCA_OUT, OLS_PREDEFINED_PCA_OUT])
    data = add_group(read_csv_numeric(PCA_PREDEFINED_OUT / "predefined_roi_pca_scores_wide.csv"))
    data["age"] = mean_impute(data["age"])
    analyses = analysis_names(data)

    for analysis in analyses:
        data[f"{analysis}_PC1_delta_adjusted"] = residualize_delta(data, analysis)

    adjusted_cols = ["subject_id", "xtc_group", "vwrec_delta", "aseg+DKT_BrainSegVol_pre"]
    for analysis in analyses:
        adjusted_cols += [f"{analysis}_PC1_pre", f"{analysis}_PC1_delta", f"{analysis}_PC1_delta_adjusted"]
    data[adjusted_cols].to_csv(CORR_PREDEFINED_PCA_OUT / "predefined_roi_pca_raw_adjusted_dataset.csv", index=False)

    corr_tables = []
    samples = [("Overall", data)] + [(g, data.loc[data.xtc_group.eq(g)]) for g in GROUP_ORDER]
    for sample_name, sample in samples:
        for adjustment in ("raw", "adjusted"):
            rows = []
            for analysis in analyses:
                xcol = f"{analysis}_PC1_delta" if adjustment == "raw" else f"{analysis}_PC1_delta_adjusted"
                subset = sample[[xcol, "vwrec_delta"]].dropna()
                rho, p = spearmanr(subset[xcol], subset["vwrec_delta"])
                rows.append({"sample": sample_name, "adjustment": adjustment, "analysis": analysis, "N": len(subset), "Spearman_rho": float(rho), "p": float(p)})
            corr_tables.append(apply_fdr(pd.DataFrame(rows)))
    all_corr = pd.concat(corr_tables, ignore_index=True)
    all_corr.to_csv(CORR_PREDEFINED_PCA_OUT / "predefined_roi_pca_raw_adjusted_spearman_all_analyses.csv", index=False)

    combined = ["Combined_shape_only", "Combined_shape_firstorder"]
    supplement_tables = []
    for sample_name in GROUP_ORDER:
        for adjustment in ("raw", "adjusted"):
            table = all_corr.loc[(all_corr["sample"] == sample_name) & (all_corr["adjustment"] == adjustment) & (all_corr["analysis"].isin(combined))].copy()
            table = apply_fdr(table.drop(columns=["FDR_q"]), output_col="FDR_q_across_2_tests")
            supplement_tables.append(table)
    pd.concat(supplement_tables, ignore_index=True).to_csv(CORR_PREDEFINED_PCA_OUT / "table_combined_pc1_raw_adjusted_group_spearman.csv", index=False)

    model_rows, coefficient_rows = [], []
    for analysis in combined:
        d = data.copy()
        # ANCOVA-style presentation: follow-up delayed recall is the outcome,
        # with baseline delayed recall retained as a covariate. The stored
        # change score is follow-up minus baseline.
        d["vwrec_followup"] = d["vwrec_pre"] + d["vwrec_delta"]
        d["delta_pc1_c"] = centered(d[f"{analysis}_PC1_delta"])
        d["pc1_pre_c"] = centered(d[f"{analysis}_PC1_pre"])
        d["vwrec_pre_c"] = centered(d["vwrec_pre"])
        d["brainseg_c"] = centered(d["aseg+DKT_BrainSegVol_pre"])
        d["log_dose_c"] = centered(d["log1p_xtc"])
        d["age_c"] = centered(d["age"])
        d["iq_c"] = centered(d["iq"])
        for clean, col in POLYSUBSTANCE_COLS.items(): d[f"{clean}_c"] = centered(d[col])
        d["sex"] = d["sex"].astype("category")
        formulas = {
            "without_polysubstance": "vwrec_followup ~ delta_pc1_c + pc1_pre_c + vwrec_pre_c + brainseg_c + log_dose_c + age_c + C(sex) + iq_c",
            "with_polysubstance": "vwrec_followup ~ delta_pc1_c + pc1_pre_c + vwrec_pre_c + brainseg_c + log_dose_c + age_c + C(sex) + iq_c + cannabis_c + tobacco_c + alcohol_c + amphetamine_c + cocaine_c",
        }
        for model_type, formula in formulas.items():
            result = smf.ols(formula, data=d).fit()
            model_rows.append({"analysis": analysis, "model_type": model_type, "N": int(result.nobs), "Delta_PC1_B": result.params["delta_pc1_c"], "Delta_PC1_SE": result.bse["delta_pc1_c"], "Delta_PC1_t": result.tvalues["delta_pc1_c"], "Delta_PC1_p": result.pvalues["delta_pc1_c"], "Model_R2": result.rsquared, "Adjusted_R2": result.rsquared_adj, "formula": formula})
            coefficient_rows.append(tidy_model_result(result, f"{analysis}_{model_type}", {"analysis": analysis, "model_type": model_type}))
            write_text(OLS_PREDEFINED_PCA_OUT / f"{analysis}_{model_type}_ols_summary.txt", result.summary().as_text())

    pd.DataFrame(model_rows).to_csv(OLS_PREDEFINED_PCA_OUT / "predefined_roi_pca_ols_model_summary.csv", index=False)
    pd.concat(coefficient_rows, ignore_index=True).to_csv(OLS_PREDEFINED_PCA_OUT / "predefined_roi_pca_ols_all_coefficients.csv", index=False)
    # Convenience files corresponding to the fully adjusted manuscript tables.
    pd.DataFrame(model_rows).query("model_type == 'with_polysubstance'").to_csv(OLS_PREDEFINED_PCA_OUT / "predefined_roi_pca_fully_adjusted_ols_key_results.csv", index=False)
    pd.concat(coefficient_rows, ignore_index=True).query("model_type == 'with_polysubstance'").to_csv(OLS_PREDEFINED_PCA_OUT / "predefined_roi_pca_fully_adjusted_ols_coefficients.csv", index=False)
    print("Predefined ROI PCA raw/adjusted Spearman and OLS analyses completed.")


if __name__ == "__main__":
    main()
