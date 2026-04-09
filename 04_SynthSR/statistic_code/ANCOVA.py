#!/usr/bin/env python3

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

INPUT_CSV = "hippocampus_thalamus_by_cohort_session.csv"

ROI_COLS = [
    "Left_Hippocampus",
    "Right_Hippocampus",
    "Left_Thalamus",
    "Right_Thalamus",
]

# change these to match your real covariate names
COVARIATES = ["age", "sex", "ICV"]

# choose one session only for ANCOVA
TARGET_SESSION = "sessie1"


def run_ancova(df: pd.DataFrame, roi: str):
    needed = ["cohort", "session", roi] + COVARIATES
    sub = df[needed].copy()
    sub = sub[sub["session"] == TARGET_SESSION].dropna()

    # make categorical variables explicit
    sub["cohort"] = sub["cohort"].astype("category")
    sub["sex"] = sub["sex"].astype("category")

    formula = f"{roi} ~ C(cohort) + age + C(sex) + ICV"
    model = smf.ols(formula, data=sub).fit()

    print(f"\n{'='*70}")
    print(f"ANCOVA for {roi} ({TARGET_SESSION})")
    print(f"{'='*70}")
    print(model.summary())

    print("\nType II ANOVA table:")
    print(anova_lm(model, typ=2))


def main():
    df = pd.read_csv(INPUT_CSV)

    for roi in ROI_COLS:
        run_ancova(df, roi)


if __name__ == "__main__":
    main()