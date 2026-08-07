"""Raw/adjusted Spearman, OLS, and incremental-value tests for whole-brain PC1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.anova import anova_lm

from config import CORR_WHOLE_BRAIN_PCA_OUT, GROUP_ORDER, OLS_WHOLE_BRAIN_PCA_OUT, PCA_WHOLE_BRAIN_OUT, POLYSUBSTANCE_COLS
from utils import add_group, centered, ensure_dirs, mean_impute, read_csv_numeric, tidy_model_result, write_text


def residualize_delta(data: pd.DataFrame) -> pd.Series:
    cols = ["PC1_delta", "PC1_pre", "aseg+DKT_BrainSegVol_pre"]
    d = data[cols].dropna()
    model = sm.OLS(d["PC1_delta"], sm.add_constant(d[["PC1_pre", "aseg+DKT_BrainSegVol_pre"]])).fit()
    out = pd.Series(np.nan, index=data.index, dtype=float)
    out.loc[d.index] = model.resid
    return out


def main() -> None:
    ensure_dirs([CORR_WHOLE_BRAIN_PCA_OUT, OLS_WHOLE_BRAIN_PCA_OUT])
    data = add_group(read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_scores_wide.csv"))
    data["age"] = mean_impute(data["age"])
    data["PC1_delta_adjusted"] = residualize_delta(data)
    data[["subject_id", "xtc_group", "PC1_pre", "PC1_delta", "PC1_delta_adjusted", "aseg+DKT_BrainSegVol_pre", "vwrec_pre", "vwrec_delta"]].to_csv(CORR_WHOLE_BRAIN_PCA_OUT / "whole_brain_pc1_raw_adjusted_dataset.csv", index=False)

    rows = []
    for sample_name, sample in [("Overall", data)] + [(g, data.loc[data.xtc_group.eq(g)]) for g in GROUP_ORDER]:
        for adjustment, xcol in [("raw", "PC1_delta"), ("adjusted_for_baseline_PC1_and_BrainSegVol", "PC1_delta_adjusted")]:
            subset = sample[[xcol, "vwrec_delta"]].dropna()
            rho, p = spearmanr(subset[xcol], subset["vwrec_delta"])
            r, pearson_p = pearsonr(subset[xcol], subset["vwrec_delta"])
            rows.append({"sample": sample_name, "adjustment": adjustment, "N": len(subset), "Spearman_rho": float(rho), "p": float(p), "FDR_q": float(p), "Pearson_r": float(r), "Pearson_p": float(pearson_p)})
    pd.DataFrame(rows).to_csv(CORR_WHOLE_BRAIN_PCA_OUT / "whole_brain_pc1_raw_adjusted_spearman.csv", index=False)

    d = data.copy()
    # ANCOVA-style presentation: follow-up delayed recall is the outcome,
    # with baseline delayed recall retained as a covariate. The stored
    # change score is follow-up minus baseline.
    d["vwrec_followup"] = d["vwrec_pre"] + d["vwrec_delta"]
    d["delta_pc1_c"] = centered(d["PC1_delta"])
    d["pc1_pre_c"] = centered(d["PC1_pre"])
    d["vwrec_pre_c"] = centered(d["vwrec_pre"])
    d["brainseg_c"] = centered(d["aseg+DKT_BrainSegVol_pre"])
    d["log_dose_c"] = centered(d["log1p_xtc"])
    d["age_c"] = centered(d["age"])
    d["iq_c"] = centered(d["iq"])
    for clean, col in POLYSUBSTANCE_COLS.items(): d[f"{clean}_c"] = centered(d[col])
    d["sex"] = d["sex"].astype("category")

    common = "vwrec_pre_c + brainseg_c + log_dose_c + age_c + C(sex) + iq_c + cannabis_c + tobacco_c + alcohol_c + amphetamine_c + cocaine_c"
    base_formula = f"vwrec_followup ~ {common}"
    full_formula = f"vwrec_followup ~ delta_pc1_c + pc1_pre_c + {common}"
    base = smf.ols(base_formula, data=d).fit()
    full = smf.ols(full_formula, data=d).fit()

    key = pd.DataFrame([{
        "N": int(full.nobs), "Delta_PC1_B": full.params["delta_pc1_c"], "Delta_PC1_SE": full.bse["delta_pc1_c"],
        "Delta_PC1_t": full.tvalues["delta_pc1_c"], "Delta_PC1_p": full.pvalues["delta_pc1_c"],
        "Model_R2": full.rsquared, "Adjusted_R2": full.rsquared_adj, "formula": full_formula,
    }])
    key.to_csv(OLS_WHOLE_BRAIN_PCA_OUT / "whole_brain_pca_fully_adjusted_ols_key_results.csv", index=False)
    tidy_model_result(full, "whole_brain_PC1", {"model_type": "full"}).to_csv(OLS_WHOLE_BRAIN_PCA_OUT / "whole_brain_pca_fully_adjusted_ols_coefficients.csv", index=False)
    write_text(OLS_WHOLE_BRAIN_PCA_OUT / "whole_brain_pca_fully_adjusted_ols_summary.txt", full.summary().as_text())
    write_text(OLS_WHOLE_BRAIN_PCA_OUT / "whole_brain_brainsegvol_only_ols_summary.txt", base.summary().as_text())

    comparison = anova_lm(base, full)
    added = pd.DataFrame([{
        "N": int(full.nobs), "base_R2": base.rsquared, "full_R2": full.rsquared,
        "delta_R2": full.rsquared - base.rsquared, "partial_F": float(comparison.iloc[1]["F"]),
        "partial_F_p": float(comparison.iloc[1]["Pr(>F)"]), "df_added": int(comparison.iloc[1]["df_diff"]),
        "base_formula": base_formula, "full_formula": full_formula,
    }])
    added.to_csv(OLS_WHOLE_BRAIN_PCA_OUT / "whole_brain_pc1_incremental_value_beyond_BrainSegVol.csv", index=False)
    print("Whole-brain PC1 raw/adjusted Spearman, OLS, and incremental-value analyses completed.")


if __name__ == "__main__":
    main()
