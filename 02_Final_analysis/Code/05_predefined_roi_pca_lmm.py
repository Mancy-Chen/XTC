"""LMMs for predefined-ROI radiomics PC1 scores."""
from __future__ import annotations

import pandas as pd

from config import LMM_PREDEFINED_PCA_OUT, PCA_PREDEFINED_OUT, POLYSUBSTANCE_COLS
from utils import centered, ensure_dirs, fit_mixedlm, mean_impute, read_csv_numeric, tidy_model_result, write_text


def analysis_names(scores: pd.DataFrame) -> list[str]:
    suffix = "_PC1_pre"
    return [c[: -len(suffix)] for c in scores.columns if c.endswith(suffix)]


def make_long(scores: pd.DataFrame, analysis: str) -> pd.DataFrame:
    meta = scores[[
        "subject_id", "sex", "age", "log1p_xtc", "aseg+DKT_BrainSegVol_pre",
        *POLYSUBSTANCE_COLS.values(),
    ]].copy()
    pre = meta.copy(); pre["time"] = 0; pre["PC1"] = scores[f"{analysis}_PC1_pre"]
    post = meta.copy(); post["time"] = 1; post["PC1"] = scores[f"{analysis}_PC1_followup"]
    long = pd.concat([pre, post], ignore_index=True)
    long["age"] = mean_impute(long["age"])
    long["age_c"] = centered(long["age"])
    long["log_dose_c"] = centered(long["log1p_xtc"])
    long["BrainSegVol_c"] = centered(long["aseg+DKT_BrainSegVol_pre"])
    for clean, col in POLYSUBSTANCE_COLS.items():
        long[f"{clean}_c"] = centered(long[col])
    long["sex"] = long["sex"].astype("category")
    return long.dropna(subset=["subject_id", "PC1", "time", "log_dose_c", "age_c", "BrainSegVol_c", "sex"])


def main() -> None:
    ensure_dirs([LMM_PREDEFINED_PCA_OUT])
    scores = read_csv_numeric(PCA_PREDEFINED_OUT / "predefined_roi_pca_scores_wide.csv")
    primary_rows = []
    sensitivity_rows = []
    key_rows = []

    for analysis in analysis_names(scores):
        long = make_long(scores, analysis)
        formula = "PC1 ~ time * log_dose_c + age_c + C(sex) + BrainSegVol_c"
        result, method = fit_mixedlm(formula, long, "subject_id")
        primary_rows.append(
            tidy_model_result(result, analysis, {"analysis": analysis, "optimizer": method, "model_type": "primary"})
        )
        term = "time:log_dose_c"
        key_rows.append({
            "analysis": analysis,
            "model_type": "primary",
            "term": term,
            "beta": result.params[term],
            "std_error": result.bse[term],
            "z": result.tvalues[term],
            "p": result.pvalues[term],
            "N_subjects": long["subject_id"].nunique(),
            "N_observations": len(long),
        })
        write_text(
            LMM_PREDEFINED_PCA_OUT / f"{analysis}_primary_summary.txt",
            result.summary().as_text(),
        )

        if analysis in {"Combined_shape_only", "Combined_shape_firstorder"}:
            sensitivity_formula = (
                formula
                + " + cannabis_c + tobacco_c + alcohol_c + amphetamine_c + cocaine_c"
            )
            sensitivity, sensitivity_method = fit_mixedlm(
                sensitivity_formula, long.dropna(), "subject_id"
            )
            sensitivity_rows.append(
                tidy_model_result(
                    sensitivity,
                    analysis,
                    {
                        "analysis": analysis,
                        "optimizer": sensitivity_method,
                        "model_type": "polysubstance_adjusted",
                    },
                )
            )
            key_rows.append({
                "analysis": analysis,
                "model_type": "polysubstance_adjusted",
                "term": term,
                "beta": sensitivity.params[term],
                "std_error": sensitivity.bse[term],
                "z": sensitivity.tvalues[term],
                "p": sensitivity.pvalues[term],
                "N_subjects": long.dropna()["subject_id"].nunique(),
                "N_observations": len(long.dropna()),
            })
            write_text(
                LMM_PREDEFINED_PCA_OUT / f"{analysis}_polysubstance_adjusted_summary.txt",
                sensitivity.summary().as_text(),
            )

    pd.concat(primary_rows, ignore_index=True).to_csv(
        LMM_PREDEFINED_PCA_OUT / "predefined_roi_pca_lmm_primary_all_terms.csv", index=False
    )
    if sensitivity_rows:
        pd.concat(sensitivity_rows, ignore_index=True).to_csv(
            LMM_PREDEFINED_PCA_OUT / "predefined_roi_pca_lmm_polysubstance_all_terms.csv", index=False
        )
    pd.DataFrame(key_rows).to_csv(
        LMM_PREDEFINED_PCA_OUT / "predefined_roi_pca_lmm_key_results.csv", index=False
    )
    print("Predefined ROI PCA LMMs completed.")


if __name__ == "__main__":
    main()
