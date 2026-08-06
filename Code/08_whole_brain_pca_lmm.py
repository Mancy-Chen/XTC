"""Primary and polysubstance-adjusted LMMs for whole-brain VoxelVolume PC1."""
from __future__ import annotations

import pandas as pd

from config import LMM_WHOLE_BRAIN_PCA_OUT, PCA_WHOLE_BRAIN_OUT, POLYSUBSTANCE_COLS
from utils import centered, ensure_dirs, fit_mixedlm, mean_impute, read_csv_numeric, tidy_model_result, write_text


def make_long(scores: pd.DataFrame) -> pd.DataFrame:
    meta = scores[[
        "subject_id", "sex", "age", "log1p_xtc", "aseg+DKT_BrainSegVol_pre",
        *POLYSUBSTANCE_COLS.values(),
    ]].copy()
    pre = meta.copy(); pre["time"] = 0; pre["PC1"] = scores["PC1_pre"]
    post = meta.copy(); post["time"] = 1; post["PC1"] = scores["PC1_followup"]
    long = pd.concat([pre, post], ignore_index=True)
    long["age"] = mean_impute(long["age"])
    long["age_c"] = centered(long["age"])
    long["log_dose_c"] = centered(long["log1p_xtc"])
    long["BrainSegVol_c"] = centered(long["aseg+DKT_BrainSegVol_pre"])
    for clean, col in POLYSUBSTANCE_COLS.items():
        long[f"{clean}_c"] = centered(long[col])
    long["sex"] = long["sex"].astype("category")
    return long.dropna()


def main() -> None:
    ensure_dirs([LMM_WHOLE_BRAIN_PCA_OUT])
    scores = read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_scores_wide.csv")
    long = make_long(scores)
    formulas = {
        "primary": "PC1 ~ time * log_dose_c + age_c + C(sex) + BrainSegVol_c",
        "polysubstance_adjusted": (
            "PC1 ~ time * log_dose_c + age_c + C(sex) + BrainSegVol_c "
            "+ cannabis_c + tobacco_c + alcohol_c + amphetamine_c + cocaine_c"
        ),
    }

    all_terms = []
    key_rows = []
    for model_name, formula in formulas.items():
        result, method = fit_mixedlm(formula, long, "subject_id")
        all_terms.append(
            tidy_model_result(
                result,
                model_name,
                {"model_type": model_name, "optimizer": method},
            )
        )
        for term in ["time", "log_dose_c", "time:log_dose_c", "BrainSegVol_c", "tobacco_c"]:
            if term in result.params.index:
                key_rows.append({
                    "model_type": model_name,
                    "term": term,
                    "beta": result.params[term],
                    "std_error": result.bse[term],
                    "z": result.tvalues[term],
                    "p": result.pvalues[term],
                    "N_subjects": long["subject_id"].nunique(),
                    "N_observations": len(long),
                })
        write_text(
            LMM_WHOLE_BRAIN_PCA_OUT / f"whole_brain_{model_name}_lmm_summary.txt",
            result.summary().as_text(),
        )

    pd.concat(all_terms, ignore_index=True).to_csv(
        LMM_WHOLE_BRAIN_PCA_OUT / "whole_brain_pca_lmm_all_terms.csv", index=False
    )
    pd.DataFrame(key_rows).to_csv(
        LMM_WHOLE_BRAIN_PCA_OUT / "whole_brain_pca_lmm_key_results.csv", index=False
    )
    print("Whole-brain PCA LMMs completed.")


if __name__ == "__main__":
    main()
