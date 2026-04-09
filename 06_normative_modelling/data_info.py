import numpy as np
import pandas as pd
from pathlib import Path

ASEG_PATH = Path("/data/projects/CSC/data/XTC/next_xtc/Normative_modeling_output/aseg_stats.xlsx")
COV_PATH  = Path("/data/projects/CSC/data/XTC/next_xtc/Normative_modeling_output/merged_all_core_with_age.xlsx")

# ---------- Load cov ----------
cov = pd.read_excel(COV_PATH)

# strip whitespace from ALL object/string columns (avoids .str misuse)
for c in cov.columns:
    if cov[c].dtype == "object":
        cov[c] = cov[c].astype(str).str.strip()

# required columns
if "studnr" not in cov.columns or "sessie" not in cov.columns:
    raise ValueError(f"Missing studnr/sessie. Columns are: {list(cov.columns)}")

cov["studnr"] = cov["studnr"].astype(str).str.strip()
cov["sessie"] = cov["sessie"].astype(str).str.strip().str.lower()

# normalize sessie to tp1/tp2/tp3 where possible
cov["sessie"] = cov["sessie"].replace({
    "1":"tp1","2":"tp2","3":"tp3",
    "tp1":"tp1","tp2":"tp2","tp3":"tp3",
    "sessie1":"tp1","sessie2":"tp2","sessie3":"tp3",
    "nan": np.nan, "none": np.nan, "": np.nan
})

print("=== COV table ===")
print("Unique subjects:", cov["studnr"].nunique(dropna=True))
print("Session rows (all):", len(cov))
print("Session rows (tp1/tp2/tp3 only):", cov[cov["sessie"].isin(["tp1","tp2","tp3"])].shape[0])
print("\nSession counts:")
print(cov["sessie"].value_counts(dropna=False))

# ---------- Load aseg ----------
aseg = pd.read_excel(ASEG_PATH, index_col=0)
aseg.index = aseg.index.astype(str).str.strip()

mask_long = aseg.index.str.contains(r"\.long\.") & aseg.index.str.contains(r"_base$")
aseg_long = aseg.loc[mask_long].copy()

# subject is prefix before first underscore: I010 / II022 / III... / IV...
aseg_long_subject = aseg_long.index.str.split("_").str[0]

print("\n=== ASEG longitudinal (*.long.*_base) ===")
print("Rows (sessions):", len(aseg_long))
print("Unique subjects:", aseg_long_subject.nunique())

# ---------- Merge counts (what your model actually uses) ----------
# build aseg_id in cov to match aseg_long index naming
cov["aseg_id"] = cov["studnr"] + "_" + cov["sessie"] + ".long." + cov["studnr"] + "_base"
cov["aseg_id"] = cov["aseg_id"].astype(str).str.strip()

aseg_long_df = aseg_long.reset_index().rename(columns={aseg_long.reset_index().columns[0]: "aseg_id"})
aseg_long_df["aseg_id"] = aseg_long_df["aseg_id"].astype(str).str.strip()

dfm = cov.merge(aseg_long_df[["aseg_id"]], on="aseg_id", how="inner")

print("\n=== MERGED (cov ∩ aseg_long) ===")
print("Rows (sessions):", len(dfm))
print("Unique subjects:", dfm["studnr"].nunique(dropna=True))
print("\nMerged session counts:")
print(dfm["sessie"].value_counts(dropna=False))

import numpy as np
import pandas as pd

# If your merged df is named dfm from the earlier snippet, use df = dfm:
df = dfm.copy()

# ---- Ensure cohort_num exists (use cohort column if numeric, else infer from studnr prefix) ----
if "cohort_num" not in df.columns:
    df["cohort_num"] = pd.to_numeric(df.get("cohort", np.nan), errors="coerce")

if df["cohort_num"].isna().all():
    def cohort_from_studnr(x):
        x = str(x)
        if x.startswith("IV"): return 4
        if x.startswith("III"): return 3
        if x.startswith("II"): return 2
        if x.startswith("I"): return 1
        return np.nan
    df["cohort_num"] = df["studnr"].apply(cohort_from_studnr)

df["cohort_num"] = pd.to_numeric(df["cohort_num"], errors="coerce")

# ---- Normalize sessie just in case ----
df["sessie"] = df["sessie"].astype(str).str.strip().str.lower().replace({
    "1":"tp1","2":"tp2","3":"tp3",
    "sessie1":"tp1","sessie2":"tp2","sessie3":"tp3"
})

# ---- Define groups ----
train = df[(df["cohort_num"] == 1) & (df["sessie"] == "tp1")]
groups = {
    "C1_tp2": df[(df["cohort_num"] == 1) & (df["sessie"] == "tp2")],
    "C1_tp3": df[(df["cohort_num"] == 1) & (df["sessie"] == "tp3")],
    "C2_tp1": df[(df["cohort_num"] == 2) & (df["sessie"] == "tp1")],
    "C3_tp1": df[(df["cohort_num"] == 3) & (df["sessie"] == "tp1")],
    "C4_tp1": df[(df["cohort_num"] == 4) & (df["sessie"] == "tp1")],
}

print("=== Group Ns (rows/sessions) ===")
print("Train (C1_tp1):", len(train))
for k, g in groups.items():
    print(f"{k}: {len(g)}")

print("\n=== Group Ns (unique subjects) ===")
print("Train (C1_tp1):", train["studnr"].nunique())
for k, g in groups.items():
    print(f"{k}: {g['studnr'].nunique()}")

print("\n=== Cohort x Session table (rows/sessions) ===")
print(pd.crosstab(df["cohort_num"], df["sessie"], dropna=False))

print("\n=== Cohort x Session table (unique subjects) ===")
tmp = df.drop_duplicates(["cohort_num", "sessie", "studnr"])
print(pd.crosstab(tmp["cohort_num"], tmp["sessie"], dropna=False))
