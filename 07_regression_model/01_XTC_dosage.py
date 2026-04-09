import pandas as pd
import numpy as np

# =========================================================
# PATHS
# =========================================================
orig_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/whole_brain_volume/merged_all.csv"   # <-- change this
out_long = "xtc_dose_long.csv"
out_wide = "xtc_dose_wide.csv"

# =========================================================
# LOAD
# =========================================================

# Try automatic separator detection with python engine
df = pd.read_csv(orig_path, sep= ';', engine="python")

print(df.shape)
print(df.columns.tolist())
print(df.head())
print("Loaded:", df.shape)


# =========================================================
# KEEP RELEVANT COLUMNS
# =========================================================
keep_cols = ["cohort", "sessie", "studnr", "xlttot"]
dose_df = df[keep_cols].copy()

# =========================================================
# CLEAN
# =========================================================
dose_df["cohort"] = pd.to_numeric(dose_df["cohort"], errors="coerce")
dose_df["sessie"] = pd.to_numeric(dose_df["sessie"], errors="coerce")
dose_df["xlttot"] = pd.to_numeric(
    dose_df["xlttot"].astype(str).str.replace(",", ".", regex=False).str.strip(),
    errors="coerce"
)

dose_df["studnr"] = dose_df["studnr"].astype(str).str.strip()

# if studnr is already like I010, keep it
# if studnr is numeric like 10, convert to I010
def make_subject_id(x):
    s = str(x).strip()
    if s.startswith("I"):
        return s
    try:
        return f"I{int(float(s)):03d}"
    except:
        return np.nan

dose_df["subject_id"] = dose_df["studnr"].apply(make_subject_id)

# =========================================================
# FILTER
# =========================================================
dose_df = dose_df[
    (dose_df["cohort"] == 1) &
    (dose_df["sessie"].isin([1, 2, 3]))
].copy()

# optional: rename session labels if you want
dose_df["session_label"] = dose_df["sessie"].map({
    1: "sessie1",
    2: "sessie2",
    3: "sessie3"
})

# =========================================================
# SORT / SAVE LONG FORMAT
# =========================================================
dose_df = dose_df.sort_values(["subject_id", "sessie"]).reset_index(drop=True)

print("\nDose long format:")
print(dose_df.head(20))

dose_df.to_csv(out_long, index=False)
print(f"Saved long file: {out_long}")

# =========================================================
# MAKE WIDE FORMAT
# =========================================================
dose_wide = dose_df.pivot_table(
    index=["subject_id", "cohort"],
    columns="sessie",
    values="xlttot",
    aggfunc="first"
).reset_index()

dose_wide = dose_wide.rename(columns={
    1: "xlttot_sessie1",
    2: "xlttot_sessie2",
    3: "xlttot_sessie3"
})

print("\nDose wide format:")
print(dose_wide.head(20))

dose_wide.to_csv(out_wide, index=False)
print(f"Saved wide file: {out_wide}")
##########################################################################################################################
# Filter the cases
import pandas as pd

# =========================================================
# PATHS
# =========================================================
dose_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/XTC_dosage/xtc_dose_wide.csv"
y_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/y_with_id_filteredQC.csv"

out_path = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/02_filteredQC/XTC_dosage/xtc_dose_wide_mapped_to_y.csv"

# =========================================================
# LOAD
# =========================================================
dose_df = pd.read_csv(dose_path)
y_df = pd.read_csv(y_path)

print("dose_df:", dose_df.shape)
print("y_df:", y_df.shape)

print("\nDose columns:", dose_df.columns.tolist())
print("Y columns:", y_df.columns.tolist())

# =========================================================
# CLEAN IDS
# =========================================================
dose_df["subject_id"] = dose_df["subject_id"].astype(str).str.strip()
y_df["subject_code"] = y_df["subject_code"].astype(str).str.strip()

# =========================================================
# MERGE
# =========================================================
mapped_df = y_df.merge(
    dose_df,
    left_on="subject_code",
    right_on="subject_id",
    how="left"
)

print("\nMapped df:", mapped_df.shape)
print(mapped_df.head())

# =========================================================
# CHECK MATCHING
# =========================================================
n_total = len(mapped_df)
n_matched = mapped_df["subject_id"].notna().sum()
n_unmatched = mapped_df["subject_id"].isna().sum()

print(f"\nTotal rows in y: {n_total}")
print(f"Matched rows: {n_matched}")
print(f"Unmatched rows: {n_unmatched}")

if n_unmatched > 0:
    print("\nUnmatched subject_code values:")
    print(mapped_df.loc[mapped_df["subject_id"].isna(), "subject_code"].drop_duplicates().tolist())

# =========================================================
# SAVE
# =========================================================
mapped_df.to_csv(out_path, index=False)
print(f"\nSaved mapped file to:\n{out_path}")