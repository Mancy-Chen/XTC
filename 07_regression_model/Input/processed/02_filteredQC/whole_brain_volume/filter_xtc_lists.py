
import os
import pandas as pd

# Paths
list_a_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/whole_brain_volume/all_stats_files_wide_volume.csv"
list_b_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/y_with_id_filteredQC.csv"
out_dir = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/whole_brain_volume"

# Read files
list_a = pd.read_csv(list_a_path)
list_b = pd.read_csv(list_b_path)

# Normalize column names
list_a.columns = [c.strip() for c in list_a.columns]
list_b.columns = [c.strip() for c in list_b.columns]

# Cohort prefix map
cohort_prefix = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV"
}

# Build subject_id from cohort + id
list_b["subject_id"] = list_b.apply(
    lambda row: f"{cohort_prefix[int(row['cohort'])]}{int(row['id']):03d}",
    axis=1
)

# Filter list A based on subject IDs in list B
filtered_a = list_a[list_a["subject_id"].isin(list_b["subject_id"])].copy()

# Save filtered full list
filtered_a_path = os.path.join(out_dir, "all_stats_files_wide_volume_filtered_by_y.csv")
filtered_a.to_csv(filtered_a_path, index=False)

# list_A_pre = sessie1 only
list_A_pre = filtered_a[filtered_a["session"].astype(str).str.strip().str.lower() == "sessie1"].copy()
list_A_pre_path = os.path.join(out_dir, "list_A_pre_sessie1.csv")
list_A_pre.to_csv(list_A_pre_path, index=False)

# list_A_delta = sessie3 - sessie1 for subjects with both sessions
session_col = "session"
subject_col = "subject_id"

# Identify non-key columns to subtract
key_cols = [subject_col, session_col]
value_cols = [c for c in filtered_a.columns if c not in key_cols]

# Split sessions
a1 = filtered_a[filtered_a[session_col].astype(str).str.strip().str.lower() == "sessie1"].copy()
a3 = filtered_a[filtered_a[session_col].astype(str).str.strip().str.lower() == "sessie3"].copy()

# Keep one row per subject per session
a1 = a1.drop_duplicates(subset=[subject_col]).set_index(subject_col)
a3 = a3.drop_duplicates(subset=[subject_col]).set_index(subject_col)

common_subjects = a1.index.intersection(a3.index)

a1_common = a1.loc[common_subjects, value_cols].apply(pd.to_numeric, errors="coerce")
a3_common = a3.loc[common_subjects, value_cols].apply(pd.to_numeric, errors="coerce")

delta_values = a3_common - a1_common
list_A_delta = delta_values.reset_index()
list_A_delta.insert(1, "session", "delta_sessie3_minus_sessie1")

list_A_delta_path = os.path.join(out_dir, "list_A_delta_sessie3_minus_sessie1.csv")
list_A_delta.to_csv(list_A_delta_path, index=False)

# list_A_pre_delta = merge pre + delta on subject_id
list_A_pre_merge = list_A_pre.drop(columns=[session_col], errors="ignore").copy()
list_A_delta_merge = list_A_delta.drop(columns=[session_col], errors="ignore").copy()

list_A_pre_delta = list_A_pre_merge.merge(
    list_A_delta_merge,
    on=subject_col,
    how="inner",
    suffixes=("_pre", "_delta")
)

list_A_pre_delta_path = os.path.join(out_dir, "list_A_pre_delta_merged.csv")
list_A_pre_delta.to_csv(list_A_pre_delta_path, index=False)

print("Saved files:")
print(filtered_a_path)
print(list_A_pre_path)
print(list_A_delta_path)
print(list_A_pre_delta_path)
