import os
import glob
import re
import pandas as pd

ROOT_DIR = "/data/projects/CSC/code/XTC/04_SynthSR/Whole_data/segment_output"
OUTPUT_DIR = "/data/projects/CSC/code/XTC/05_Radiomics_Mancy"

LONG_CSV = os.path.join(OUTPUT_DIR, "all_stats_files_long.csv")
WIDE_CSV = os.path.join(OUTPUT_DIR, "all_stats_files_wide_volume.csv")

MEASURE_LONG_CSV = os.path.join(OUTPUT_DIR, "all_stats_files_measures_long.csv")
MEASURE_WIDE_CSV = os.path.join(OUTPUT_DIR, "all_stats_files_measures_wide.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_subject_session(folder_name):
    """
    Example:
    I010_sessie1 -> subject_id=I010, session=sessie1
    """
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return folder_name, None


def convert_value(x):
    x = str(x).strip()
    try:
        if re.fullmatch(r"[+-]?\d+", x):
            return int(x)
        return float(x)
    except Exception:
        return x


def parse_measure_line(line):
    """
    Example:
    # Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1439247.833200, mm^3
    """
    line = line.strip()
    if not line.startswith("# Measure "):
        return None

    content = line[len("# Measure "):]
    parts = [p.strip() for p in content.split(",")]

    if len(parts) < 5:
        return None

    return {
        "measure_name": parts[0],
        "measure_key": parts[1],
        "measure_description": parts[2],
        "measure_value": convert_value(parts[3]),
        "measure_unit": parts[4],
    }


def parse_stats_file(stats_path):
    with open(stats_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    colheaders = None
    table_rows = []
    measure_rows = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# Measure "):
            m = parse_measure_line(stripped)
            if m is not None:
                measure_rows.append(m)

        if stripped.startswith("# ColHeaders"):
            colheaders = stripped.replace("# ColHeaders", "", 1).strip().split()
            continue

        if colheaders is not None:
            if stripped == "" or stripped.startswith("#"):
                continue

            parts = stripped.split()

            if len(parts) != len(colheaders):
                continue

            row = {}
            for c, v in zip(colheaders, parts):
                row[c] = convert_value(v)
            table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    measure_df = pd.DataFrame(measure_rows)

    return table_df, measure_df


all_table_rows = []
all_measure_rows = []

stats_files = sorted(glob.glob(os.path.join(ROOT_DIR, "*", "stats", "*.stats")))
print(f"Found {len(stats_files)} .stats files")

for stats_path in stats_files:
    try:
        subject_folder = os.path.basename(os.path.dirname(os.path.dirname(stats_path)))
        stats_file = os.path.basename(stats_path)

        subject_id, session = parse_subject_session(subject_folder)

        table_df, measure_df = parse_stats_file(stats_path)

        if not table_df.empty:
            table_df.insert(0, "subject_folder", subject_folder)
            table_df.insert(1, "subject_id", subject_id)
            table_df.insert(2, "session", session)
            table_df.insert(3, "stats_file", stats_file)
            table_df.insert(4, "stats_path", stats_path)
            all_table_rows.append(table_df)

        if not measure_df.empty:
            measure_df.insert(0, "subject_folder", subject_folder)
            measure_df.insert(1, "subject_id", subject_id)
            measure_df.insert(2, "session", session)
            measure_df.insert(3, "stats_file", stats_file)
            measure_df.insert(4, "stats_path", stats_path)
            all_measure_rows.append(measure_df)

        print(f"[OK] Parsed {stats_file} for {subject_folder}")

    except Exception as e:
        print(f"[ERROR] Failed {stats_path}: {e}")

# =========================================================
# LONG ROI TABLE
# =========================================================
if all_table_rows:
    long_df = pd.concat(all_table_rows, axis=0, ignore_index=True)
    long_df.to_csv(LONG_CSV, index=False)
    print(f"\nSaved long ROI table: {LONG_CSV}")
    print(f"Rows: {len(long_df)}")
else:
    long_df = pd.DataFrame()
    print("\nNo ROI table rows found.")

# =========================================================
# WIDE ROI TABLE (Volume only)
# one row per subject_id + session
# one column per stats_file + StructName
# =========================================================
if not long_df.empty:
    if "StructName" not in long_df.columns or "Volume_mm3" not in long_df.columns:
        raise ValueError("Expected columns StructName and Volume_mm3 not found in ROI table.")

    wide_df = long_df.pivot_table(
        index=["subject_id", "session"],
        columns=["stats_file", "StructName"],
        values="Volume_mm3",
        aggfunc="first"
    ).reset_index()

    wide_df.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).replace(".stats", "")
        if isinstance(col, tuple) else col
        for col in wide_df.columns
    ]

    wide_df.to_csv(WIDE_CSV, index=False)
    print(f"Saved wide ROI volume table: {WIDE_CSV}")
    print(f"Rows: {len(wide_df)}")

# =========================================================
# LONG MEASURE TABLE
# =========================================================
if all_measure_rows:
    measure_long_df = pd.concat(all_measure_rows, axis=0, ignore_index=True)
    measure_long_df.to_csv(MEASURE_LONG_CSV, index=False)
    print(f"\nSaved long measure table: {MEASURE_LONG_CSV}")
    print(f"Rows: {len(measure_long_df)}")
else:
    measure_long_df = pd.DataFrame()
    print("\nNo Measure rows found.")

# =========================================================
# WIDE MEASURE TABLE
# one row per subject_id + session
# one column per stats_file + measure_key
# =========================================================
if not measure_long_df.empty:
    if "measure_key" not in measure_long_df.columns or "measure_value" not in measure_long_df.columns:
        raise ValueError("Expected columns measure_key and measure_value not found in measure table.")

    measure_wide_df = measure_long_df.pivot_table(
        index=["subject_id", "session"],
        columns=["stats_file", "measure_key"],
        values="measure_value",
        aggfunc="first"
    ).reset_index()

    measure_wide_df.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).replace(".stats", "")
        if isinstance(col, tuple) else col
        for col in measure_wide_df.columns
    ]

    measure_wide_df.to_csv(MEASURE_WIDE_CSV, index=False)
    print(f"Saved wide measure table: {MEASURE_WIDE_CSV}")
    print(f"Rows: {len(measure_wide_df)}")

print("\nDone.")