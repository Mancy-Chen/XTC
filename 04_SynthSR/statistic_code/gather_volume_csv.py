#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from pathlib import Path

BASE_DIR = Path("/data/projects/CSC/code/XTC/04_SynthSR/Whole_data/segment_output")
OUTPUT_CSV = Path("hippocampus_thalamus_with_globalvolumes.csv")

TARGETS = {
    "Left-Hippocampus": "Left_Hippocampus",
    "Right-Hippocampus": "Right_Hippocampus",
    "Left-Thalamus": "Left_Thalamus",
    "Right-Thalamus": "Right_Thalamus",
}

HEADER_MEASURES = {
    "MaskVol": "MaskVol",
    "BrainSegVol": "BrainSegVol",
    "BrainSegVolNotVent": "BrainSegVolNotVent",
    "SupraTentorialVol": "SupraTentorialVol",
    "SupraTentorialVolNotVent": "SupraTentorialVolNotVent",
    "SubCortGrayVol": "SubCortGrayVol",
}

FOLDER_PATTERN = re.compile(
    r"^(I{1,3}|IV)(\d+)_(?:sessie|sissie)([123])$",
    re.IGNORECASE
)


def parse_stats_file(stats_file: Path) -> dict[str, float | None]:
    results = {v: None for v in TARGETS.values()}
    results.update({v: None for v in HEADER_MEASURES.values()})

    with stats_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Parse header measures, e.g.
            # # Measure Mask, MaskVol, Mask Volume, 1566499.000000, mm^3
            if line.startswith("# Measure"):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4:
                    measure_key = parts[1]
                    if measure_key in HEADER_MEASURES:
                        out_name = HEADER_MEASURES[measure_key]
                        try:
                            results[out_name] = float(parts[3])
                        except ValueError:
                            results[out_name] = None
                continue

            # Parse ROI table rows
            if line.startswith("#"):
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 5:
                continue

            structure_name = parts[4]
            if structure_name in TARGETS:
                out_name = TARGETS[structure_name]
                try:
                    results[out_name] = float(parts[3])
                except ValueError:
                    results[out_name] = None

    return results


def find_stats_file(subject_dir: Path) -> Path | None:
    for f in [
        subject_dir / "stats" / "aseg+DKT.stats",
        subject_dir / "stats" / "aseg.stats",
    ]:
        if f.exists():
            return f
    return None


def main() -> None:
    rows = []

    for folder in sorted(BASE_DIR.iterdir()):
        if not folder.is_dir():
            continue

        m = FOLDER_PATTERN.match(folder.name)
        if not m:
            continue

        cohort = m.group(1).upper()
        numeric_id = m.group(2)
        session_num = m.group(3)

        subject_id = f"{cohort}{numeric_id}"
        session = f"sessie{session_num}"

        stats_file = find_stats_file(folder)
        if stats_file is None:
            continue

        data = parse_stats_file(stats_file)

        rows.append({
            "folder_name": folder.name,
            "cohort": cohort,
            "subject_id": subject_id,
            "session": session,
            **data,
        })

    fieldnames = [
        "folder_name",
        "cohort",
        "subject_id",
        "session",
        "Left_Hippocampus",
        "Right_Hippocampus",
        "Left_Thalamus",
        "Right_Thalamus",
        "MaskVol",
        "BrainSegVol",
        "BrainSegVolNotVent",
        "SupraTentorialVol",
        "SupraTentorialVolNotVent",
        "SubCortGrayVol",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()