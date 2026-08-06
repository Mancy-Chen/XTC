"""Longitudinal LMMs for predefined ROI VoxelVolume and BrainSegVol."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BRAINSEGVOL_INPUT, GROUP_ORDER, LMM_ROI_VOLUME_OUT, ROI_VOLUME_COLUMNS, WHOLE_BRAIN_INPUT
from utils import add_group, apply_fdr, ensure_dirs, fit_mixedlm, read_csv_numeric, tidy_model_result, write_text


def roi_long(data: pd.DataFrame, base: str) -> pd.DataFrame:
    pre = pd.to_numeric(data[f"{base}_pre"], errors="coerce")
    delta = pd.to_numeric(data[f"{base}_delta"], errors="coerce")
    brain_pre = pd.to_numeric(data["aseg+DKT_BrainSegVol_pre"], errors="coerce")  # cm3
    brain_delta = pd.to_numeric(data["aseg+DKT_BrainSegVol_delta"], errors="coerce") / 1000.0
    long = pd.DataFrame({
        "subject_id": np.repeat(data["subject_id"].to_numpy(), 2),
        "sex": np.repeat(data["sex"].to_numpy(), 2),
        "xtc_group": np.repeat(data["xtc_group"].to_numpy(), 2),
        "xtc_user": np.repeat(data["xtc_user"].to_numpy(), 2),
        "time": np.tile([0, 1], len(data)),
        "outcome_cm3": np.column_stack([pre, pre + delta]).ravel(),
        "BrainSegVol_cm3": np.column_stack([brain_pre, brain_pre + brain_delta]).ravel(),
    }).dropna()
    long["BrainSegVol_c"] = long["BrainSegVol_cm3"] - long["BrainSegVol_cm3"].mean()
    long["sex"] = long["sex"].astype("category")
    return long


def brain_long(data: pd.DataFrame) -> pd.DataFrame:
    pre = pd.to_numeric(data["aseg+DKT_BrainSegVol_pre_mm3"], errors="coerce") / 1000.0
    delta = pd.to_numeric(data["aseg+DKT_BrainSegVol_delta"], errors="coerce") / 1000.0
    long = pd.DataFrame({
        "subject_id": np.repeat(data["subject_id"].to_numpy(), 2),
        "sex": np.repeat(data["sex"].to_numpy(), 2),
        "xtc_group": np.repeat(data["xtc_group"].to_numpy(), 2),
        "xtc_user": np.repeat(data["xtc_user"].to_numpy(), 2),
        "time": np.tile([0, 1], len(data)),
        "outcome_cm3": np.column_stack([pre, pre + delta]).ravel(),
    }).dropna()
    long["sex"] = long["sex"].astype("category")
    return long


def key_row(result, term: str, analysis: str, outcome: str, n: int, optimizer: str) -> dict:
    ci = result.conf_int().loc[term]
    return {
        "analysis": analysis, "outcome": outcome, "term": term, "N": n,
        "estimate_cm3": float(result.params[term]), "std_error": float(result.bse[term]),
        "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1]),
        "z": float(result.tvalues[term]), "p": float(result.pvalues[term]),
        "optimizer": optimizer, "converged": bool(getattr(result, "converged", True)),
    }


def main() -> None:
    ensure_dirs([LMM_ROI_VOLUME_OUT])
    rad = read_csv_numeric(WHOLE_BRAIN_INPUT)
    brain = read_csv_numeric(BRAINSEGVOL_INPUT).rename(columns={"aseg+DKT_BrainSegVol_pre": "aseg+DKT_BrainSegVol_pre_mm3"})
    data = add_group(rad.merge(brain, on="subject_id", how="inner", validate="one_to_one"))

    full_tables, within_tables, interaction_rows = [], [], []
    for group_name in GROUP_ORDER:
        group_rows = []
        subset = data.loc[data["xtc_group"] == group_name].copy()
        for outcome, base in ROI_VOLUME_COLUMNS.items():
            long = roi_long(subset, base)
            result, method = fit_mixedlm("outcome_cm3 ~ time + C(sex) + BrainSegVol_c", long, "subject_id")
            group_rows.append(key_row(result, "time", f"within_{group_name}", outcome, long.subject_id.nunique(), method))
            full_tables.append(tidy_model_result(result, f"within_{group_name}_{outcome}", {"outcome": outcome, "group": group_name, "optimizer": method}))
            write_text(LMM_ROI_VOLUME_OUT / f"within_{group_name}_{outcome}_summary.txt".replace(" ", "_"), result.summary().as_text())
        long = brain_long(subset)
        result, method = fit_mixedlm("outcome_cm3 ~ time + C(sex)", long, "subject_id")
        group_rows.append(key_row(result, "time", f"within_{group_name}", "BrainSegVol", long.subject_id.nunique(), method))
        full_tables.append(tidy_model_result(result, f"within_{group_name}_BrainSegVol", {"outcome": "BrainSegVol", "group": group_name, "optimizer": method}))
        write_text(LMM_ROI_VOLUME_OUT / f"within_{group_name}_BrainSegVol_summary.txt".replace(" ", "_"), result.summary().as_text())
        group_table = apply_fdr(pd.DataFrame(group_rows))
        within_tables.append(group_table)

    for outcome, base in ROI_VOLUME_COLUMNS.items():
        long = roi_long(data, base)
        result, method = fit_mixedlm("outcome_cm3 ~ time * xtc_user + C(sex) + BrainSegVol_c", long, "subject_id")
        interaction_rows.append(key_row(result, "time:xtc_user", "between_group_change_difference", outcome, long.subject_id.nunique(), method))
        full_tables.append(tidy_model_result(result, f"interaction_{outcome}", {"outcome": outcome, "group": "all", "optimizer": method}))
        write_text(LMM_ROI_VOLUME_OUT / f"interaction_{outcome}_summary.txt".replace(" ", "_"), result.summary().as_text())
    long = brain_long(data)
    result, method = fit_mixedlm("outcome_cm3 ~ time * xtc_user + C(sex)", long, "subject_id")
    interaction_rows.append(key_row(result, "time:xtc_user", "between_group_change_difference", "BrainSegVol", long.subject_id.nunique(), method))
    full_tables.append(tidy_model_result(result, "interaction_BrainSegVol", {"outcome": "BrainSegVol", "group": "all", "optimizer": method}))
    write_text(LMM_ROI_VOLUME_OUT / "interaction_BrainSegVol_summary.txt", result.summary().as_text())

    within = pd.concat(within_tables, ignore_index=True)
    interaction = apply_fdr(pd.DataFrame(interaction_rows))
    pd.concat([within, interaction], ignore_index=True).to_csv(LMM_ROI_VOLUME_OUT / "roi_voxelvolume_lmm_key_results.csv", index=False)
    within.to_csv(LMM_ROI_VOLUME_OUT / "table_within_group_change.csv", index=False)
    interaction.to_csv(LMM_ROI_VOLUME_OUT / "table_direct_group_change_difference.csv", index=False)
    pd.concat(full_tables, ignore_index=True).to_csv(LMM_ROI_VOLUME_OUT / "roi_voxelvolume_lmm_all_coefficients.csv", index=False)
    print("Predefined ROI and BrainSegVol longitudinal LMMs completed.")


if __name__ == "__main__":
    main()
