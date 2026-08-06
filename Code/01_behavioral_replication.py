"""Replicate the previous NeXT RAVLT findings in the final imaging sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from config import BEHAVIOR_INPUT, OLS_BEHAVIOR_OUT
from utils import ensure_dirs, read_csv_numeric, tidy_model_result, write_text


def main() -> None:
    ensure_dirs([OLS_BEHAVIOR_OUT])
    df = read_csv_numeric(BEHAVIOR_INPUT)
    df["sex"] = df["sex"].astype("category")

    covariates = [
        "age",
        "dart_iq",
        "log1p_cannabis",
        "log1p_tobacco",
        "log1p_alcohol",
        "log1p_amphetamine",
        "log1p_cocaine",
    ]

    outcome_specs = {
        "Immediate recall": ("ravlt_immediate_change", "vwsom", "bvwsom", "ols"),
        "Delayed recall": ("ravlt_delayed_change", "vwrec", "bvwrec", "ols"),
        "Recognition decline": (
            "ravlt_recognition_decline",
            "vwherk",
            "bvwherk",
            "logit",
        ),
    }

    descriptive_rows = []
    coefficient_tables = []
    summaries = []

    for label, (outcome, baseline, followup, model_type) in outcome_specs.items():
        formula = (
            f"{outcome} ~ xtc_exposure + C(sex) + {baseline} + "
            + " + ".join(covariates)
        )
        model_data = df[[outcome, baseline, followup, "xtc_exposure", "sex"] + covariates].dropna().copy()

        for exposed, group_name in [(0, "XTC-naive"), (1, "XTC users")]:
            group = model_data.loc[model_data["xtc_exposure"] == exposed]
            descriptive_rows.append(
                {
                    "outcome": label,
                    "group": group_name,
                    "N": len(group),
                    "baseline_mean": group[baseline].mean(),
                    "baseline_sd": group[baseline].std(ddof=1),
                    "followup_mean": group[followup].mean(),
                    "followup_sd": group[followup].std(ddof=1),
                    "change_mean": group[outcome].mean(),
                    "change_sd": group[outcome].std(ddof=1),
                }
            )

        if model_type == "ols":
            result = smf.ols(formula, data=model_data).fit()
            anova = anova_lm(result, typ=2)
            ss_effect = float(anova.loc["xtc_exposure", "sum_sq"])
            ss_error = float(anova.loc["Residual", "sum_sq"])
            partial_eta2 = ss_effect / (ss_effect + ss_error)
            effect = float(result.params["xtc_exposure"])
            effect_type = "adjusted mean difference"
            odds_ratio = np.nan
        else:
            result = smf.glm(
                formula,
                data=model_data,
                family=sm.families.Binomial(),
            ).fit()
            partial_eta2 = np.nan
            effect = float(result.params["xtc_exposure"])
            effect_type = "log odds"
            odds_ratio = float(np.exp(effect))

        conf = result.conf_int().loc["xtc_exposure"]
        summaries.append(
            {
                "outcome": label,
                "model": model_type,
                "N": int(result.nobs),
                "adjusted_effect": effect,
                "effect_type": effect_type,
                "ci_low": float(conf.iloc[0]),
                "ci_high": float(conf.iloc[1]),
                "p": float(result.pvalues["xtc_exposure"]),
                "partial_eta_squared": partial_eta2,
                "odds_ratio": odds_ratio,
                "odds_ratio_ci_low": float(np.exp(conf.iloc[0])) if model_type == "logit" else np.nan,
                "odds_ratio_ci_high": float(np.exp(conf.iloc[1])) if model_type == "logit" else np.nan,
                "formula": formula,
            }
        )
        coefficient_tables.append(
            tidy_model_result(result, f"behavior_{label.lower().replace(' ', '_')}")
        )
        write_text(
            OLS_BEHAVIOR_OUT / f"{label.lower().replace(' ', '_')}_model_summary.txt",
            result.summary().as_text(),
        )

    pd.DataFrame(descriptive_rows).to_csv(
        OLS_BEHAVIOR_OUT / "behavioral_group_descriptives.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        OLS_BEHAVIOR_OUT / "behavioral_replication_key_results.csv", index=False
    )
    pd.concat(coefficient_tables, ignore_index=True).to_csv(
        OLS_BEHAVIOR_OUT / "behavioral_replication_full_coefficients.csv", index=False
    )
    print("Behavioral replication completed.")


if __name__ == "__main__":
    main()
