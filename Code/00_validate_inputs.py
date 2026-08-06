"""Validate canonical inputs, subject alignment, transformations, and I221 correction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BEHAVIOR_INPUT, BRAINSEGVOL_INPUT, IMAGING_COVARIATES_INPUT, LOG_DIR,
    MERGED_ALL_SAV, POLYSUBSTANCE_COLS, PREDEFINED_ROI_INPUT,
    SPATIAL_ATLAS_PATH, WHOLE_BRAIN_INPUT,
)
from utils import ensure_dirs, read_csv_numeric, require_columns, validate_unique_subjects


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ensure_dirs([LOG_DIR])
    specifications = {
        "predefined_roi": (PREDEFINED_ROI_INPUT, ["subject_id", "sex", "age", "xlttot_sessie3", "log1p_xtc", "vwrec_pre", "vwrec_delta"]),
        "whole_brain": (WHOLE_BRAIN_INPUT, ["subject_id", "sex", "age", "xlttot_sessie3", "log1p_xtc", "vwrec_pre", "vwrec_delta"]),
        "imaging_covariates": (IMAGING_COVARIATES_INPUT, ["subject_id", "sex", "age", "iq", "log1p_xtc", "vwrec_pre", "vwrec_delta"]),
        "brainsegvol": (BRAINSEGVOL_INPUT, ["subject_id", "aseg+DKT_BrainSegVol_pre", "aseg+DKT_BrainSegVol_delta"]),
        "behavior": (BEHAVIOR_INPUT, ["studnr", "xtc_exposure", "vwrec", "bvwrec", "ravlt_delayed_change"]),
    }

    report, subject_sets, loaded = {}, {}, {}
    for name, (path, required) in specifications.items():
        df = read_csv_numeric(path)
        loaded[name] = df
        require_columns(df, required, name)
        subject_col = "studnr" if name == "behavior" else "subject_id"
        validate_unique_subjects(df.rename(columns={subject_col: "subject_id"}), "subject_id")
        ids = set(df[subject_col].astype(str).str.strip().str.upper())
        subject_sets[name] = ids
        report[name] = {
            "path": str(path.relative_to(path.parents[2])),
            "rows": int(df.shape[0]), "columns": int(df.shape[1]),
            "unique_subjects": len(ids), "sha256": sha256(path),
        }

    reference = subject_sets["imaging_covariates"]
    for name in subject_sets:
        report[name]["same_subjects_as_imaging_covariates"] = subject_sets[name] == reference
        report[name]["missing_from_reference"] = sorted(reference - subject_sets[name])
        report[name]["extra_vs_reference"] = sorted(subject_sets[name] - reference)
    if any(len(ids) != 95 for ids in subject_sets.values()) or not all(ids == reference for ids in subject_sets.values()):
        raise ValueError("Canonical inputs must contain exactly the same 95 unique participants.")

    # I221 is authoritative from the original SPSS work file: VWREC=11, BVWREC=13, delta=+2.
    i221_checks = {}
    for name in ("predefined_roi", "whole_brain", "imaging_covariates"):
        row = loaded[name].loc[loaded[name]["subject_id"].eq("I221")].iloc[0]
        i221_checks[name] = {"vwrec_pre": float(row["vwrec_pre"]), "vwrec_delta": float(row["vwrec_delta"])}
        if not (float(row["vwrec_pre"]) == 11 and float(row["vwrec_delta"]) == 2):
            raise ValueError(f"I221 is not corrected in {name}: {i221_checks[name]}")
    brow = loaded["behavior"].loc[loaded["behavior"]["studnr"].eq("I221")].iloc[0]
    i221_checks["behavior"] = {"vwrec": float(brow["vwrec"]), "bvwrec": float(brow["bvwrec"]), "delta": float(brow["ravlt_delayed_change"])}
    if not (float(brow["vwrec"]) == 11 and float(brow["bvwrec"]) == 13 and float(brow["ravlt_delayed_change"]) == 2):
        raise ValueError(f"I221 is inconsistent in behavioral input: {i221_checks['behavior']}")

    # Check that canonical log columns are ln(1+x), where raw columns are available.
    transform_checks = {}
    for name in ("predefined_roi", "whole_brain", "imaging_covariates"):
        df = loaded[name]
        max_xtc_error = float(np.nanmax(np.abs(df["log1p_xtc"] - np.log1p(df["xlttot_sessie3"]))))
        transform_checks[f"{name}_xtc_max_abs_error"] = max_xtc_error
        if max_xtc_error > 1e-9:
            raise ValueError(f"XTC log transformation mismatch in {name}")

    if not MERGED_ALL_SAV.exists():
        raise FileNotFoundError(f"Required demographic source not found: {MERGED_ALL_SAV}")
    report["merged_all_sav"] = {
        "path": str(MERGED_ALL_SAV.relative_to(MERGED_ALL_SAV.parents[2])),
        "size_bytes": MERGED_ALL_SAV.stat().st_size,
        "sha256": sha256(MERGED_ALL_SAV),
    }
    report["spatial_atlas"] = {
        "configured_path": str(SPATIAL_ATLAS_PATH),
        "accessible_in_current_runtime": SPATIAL_ATLAS_PATH.exists(),
        "required_for": "PC1-PC5 anatomical loading maps only",
    }

    report["I221_correction"] = i221_checks
    report["transformation_checks"] = transform_checks

    (LOG_DIR / "input_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({k: v for k, v in report.items() if isinstance(v, dict) and "rows" in v}).T.to_csv(LOG_DIR / "input_manifest.csv")
    print("Input validation passed: 95 aligned subjects; I221 VWREC 11 -> 13 (delta +2); merged_all.sav present.")
    if not SPATIAL_ATLAS_PATH.exists():
        print(f"Optional spatial atlas is not mounted here: {SPATIAL_ATLAS_PATH}")


if __name__ == "__main__":
    main()
