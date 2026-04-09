#!/usr/bin/env python3

from __future__ import annotations

import pandas as pd
from pathlib import Path

# =========================
# INPUT / OUTPUT
# =========================
INPUT_CSV = Path("hippocampus_thalamus_by_cohort_session.csv")
OUTPUT_CSV = Path("hippocampus_thalamus_summary_by_cohort_session.csv")

ROI_COLS = [
    "Left_Hippocampus",
    "Right_Hippocampus",
    "Left_Thalamus",
    "Right_Thalamus",
]

GROUP_COLS = ["cohort", "session"]


def summarize_group(df: pd.DataFrame) -> pd.Series:
    out = {}

    for roi in ROI_COLS:
        vals = pd.to_numeric(df[roi], errors="coerce").dropna()

        if len(vals) == 0:
            out[f"{roi}_N"] = 0
            out[f"{roi}_Mean"] = None
            out[f"{roi}_Median"] = None
            out[f"{roi}_SD"] = None
            out[f"{roi}_Mean_minus_1SD"] = None
            out[f"{roi}_Mean_plus_1SD"] = None
            out[f"{roi}_Mean_minus_2SD"] = None
            out[f"{roi}_Mean_plus_2SD"] = None
            continue

        mean_val = vals.mean()
        median_val = vals.median()
        sd_val = vals.std(ddof=1) if len(vals) > 1 else 0.0

        out[f"{roi}_N"] = len(vals)
        out[f"{roi}_Mean"] = mean_val
        out[f"{roi}_Median"] = median_val
        out[f"{roi}_SD"] = sd_val
        out[f"{roi}_Mean_minus_1SD"] = mean_val - sd_val
        out[f"{roi}_Mean_plus_1SD"] = mean_val + sd_val
        out[f"{roi}_Mean_minus_2SD"] = mean_val - 2 * sd_val
        out[f"{roi}_Mean_plus_2SD"] = mean_val + 2 * sd_val

    return pd.Series(out)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    missing_cols = [c for c in GROUP_COLS + ROI_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    summary_df = (
        df.groupby(GROUP_COLS, dropna=False)
          .apply(summarize_group)
          .reset_index()
    )

    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved summary to: {OUTPUT_CSV}")

    # also print to screen
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df)


if __name__ == "__main__":
    main()