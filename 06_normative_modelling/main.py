import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf

# =========================
# Paths
# =========================
ASEG_PATH = Path(r"/data/projects/CSC/data/XTC/next_xtc/Normative_modeling_output/aseg_stats.xlsx")
COV_PATH  = Path(r"/data/projects/CSC/data/XTC/next_xtc/Normative_modeling_output/merged_all_core_with_age.xlsx")
OUT_XLSX  = Path(r"/data/projects/CSC/data/XTC/next_xtc/Normative_modeling_output/normative_results_statsmodels.xlsx")
PLOT_DIR  = Path(r"/data/projects/CSC/data/XTC/next_xtc/Normative_modeling_output/normative_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Settings
# =========================
roi_cols = ["Left-Hippocampus", "Right-Hippocampus", "Left-Thalamus", "Right-Thalamus"]
icv_col  = "EstimatedTotalIntraCranialVol"
X_cols   = ["sex_bin", "age_imp", "age2", icv_col]

PRINT_FORMULA = True   # or False
ALPHA = 0.05           # Tukey alpha (if you want it explicit here too)

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def sanitize_sheet(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)[:31]

# =========================
# 1) Load aseg (volumes)
# =========================
aseg = pd.read_excel(ASEG_PATH, index_col=0)
aseg.index = aseg.index.astype(str).str.strip()

mask_long = aseg.index.str.contains(r"\.long\.", regex=True) & aseg.index.str.contains(r"_base$", regex=True)
aseg_long = aseg.loc[mask_long].copy()

missing_cols = [c for c in roi_cols + [icv_col] if c not in aseg_long.columns]
if missing_cols:
    raise ValueError(f"Missing columns in aseg_stats.xlsx: {missing_cols}")

aseg_long = aseg_long[roi_cols + [icv_col]].apply(to_num)
aseg_long = aseg_long.reset_index()
aseg_long = aseg_long.rename(columns={aseg_long.columns[0]: "aseg_id"})
aseg_long["aseg_id"] = aseg_long["aseg_id"].astype(str).str.strip()

print("aseg_long rows:", len(aseg_long))

# =========================
# 2) Load covariates
# =========================
cov = pd.read_excel(COV_PATH)
cov.columns = [c.strip() for c in cov.columns]

required = ["cohort", "sessie", "studnr", "sex", "age"]
miss = [c for c in required if c not in cov.columns]
if miss:
    raise ValueError(f"Missing expected columns in merged_all_core_with_age.xlsx: {miss}")

cov["studnr"] = cov["studnr"].astype(str).str.strip()

# sessie -> tp1/tp2/tp3 (robust)
cov["sessie"] = cov["sessie"].astype(str).str.strip().str.lower()
cov["sessie"] = cov["sessie"].replace({
    "1":"tp1","2":"tp2","3":"tp3",
    "tp1":"tp1","tp2":"tp2","tp3":"tp3",
    "sessie1":"tp1","sessie2":"tp2","sessie3":"tp3",
})
cov.loc[cov["sessie"].isin(["", "nan", "none"]), "sessie"] = np.nan

# cohort -> numeric 1..4 if possible
cov["cohort"] = cov["cohort"].astype(str).str.strip()
cov["cohort_num"] = to_num(cov["cohort"])  # if already 1/2/3/4 this will work

# sex -> binary
s = cov["sex"].astype(str).str.strip().str.lower()
cov["sex_bin"] = s.replace({
    "m": 1, "male": 1, "man": 1, "1": 1,
    "f": 0, "female": 0, "vrouw": 0, "2": 0
})
cov["sex_bin"] = to_num(cov["sex_bin"])

# age numeric + within-subject imputation
cov["age"] = to_num(cov["age"])
cov["age_imp"] = cov["age"].fillna(cov.groupby("studnr")["age"].transform("mean"))
print("Age missing after within-subject imputation:", int(cov["age_imp"].isna().sum()))

# Build join key
cov["aseg_id"] = cov["studnr"] + "_" + cov["sessie"] + ".long." + cov["studnr"] + "_base"
cov["aseg_id"] = cov["aseg_id"].astype(str).str.strip()

# =========================
# 3) Merge
# =========================
df = cov.merge(aseg_long, on="aseg_id", how="inner")
print(f"Merged rows: {len(df)} (cov={len(cov)}, aseg_long={len(aseg_long)})")

# Normalize again AFTER merge (important)
df["sessie"] = df["sessie"].astype(str).str.strip().str.lower()
df["sessie"] = df["sessie"].replace({"1":"tp1","2":"tp2","3":"tp3","sessie1":"tp1","sessie2":"tp2","sessie3":"tp3"})

# Use cohort_num if available; else try to parse from studnr prefix (I/II/III/IV)
df["cohort_num"] = df.get("cohort_num", np.nan)
if df["cohort_num"].isna().all():
    # fallback: derive from studnr prefix
    # I -> 1, II -> 2, III -> 3, IV -> 4
    def cohort_from_studnr(x):
        x = str(x)
        if x.startswith("IV"): return 4
        if x.startswith("III"): return 3
        if x.startswith("II"): return 2
        if x.startswith("I"): return 1
        return np.nan
    df["cohort_num"] = df["studnr"].apply(cohort_from_studnr)

df["cohort_num"] = to_num(df["cohort_num"]).astype("Int64")

# Drop missing key covariates / ICV
df = df.dropna(subset=["cohort_num", "sessie", "studnr", "sex_bin", "age_imp", icv_col])
df["age2"] = df["age_imp"] ** 2

print("\nPost-merge sessie counts:")
print(df["sessie"].value_counts(dropna=False))
print("\nPost-merge cohort_num counts:")
print(df["cohort_num"].value_counts(dropna=False).sort_index())

# =========================
# 4) Train + groups
# =========================
train = df[(df["cohort_num"] == 1) & (df["sessie"] == "tp1")].copy()

groups = {
    "C1_tp2": df[(df["cohort_num"] == 1) & (df["sessie"] == "tp2")].copy(),
    "C1_tp3": df[(df["cohort_num"] == 1) & (df["sessie"] == "tp3")].copy(),
    "C2_tp1": df[(df["cohort_num"] == 2) & (df["sessie"] == "tp1")].copy(),
    "C3_tp1": df[(df["cohort_num"] == 3) & (df["sessie"] == "tp1")].copy(),
    "C4_tp1": df[(df["cohort_num"] == 4) & (df["sessie"] == "tp1")].copy(),
}

print("\nTrain N:", len(train))
for k, g in groups.items():
    print(k, "N:", len(g))

if len(train) == 0:
    raise RuntimeError(
        "Train set is empty. Check df['cohort_num'] and df['sessie'] values above. "
        "If cohort_num isn't 1..4 or sessie isn't tp1/tp2/tp3, adjust the mapping."
    )

# =========================
# 5) Fit + score (statsmodels OLS)
# =========================
def fit_model(train_df, ycol):
    X = sm.add_constant(train_df[X_cols], has_constant="add")
    y = train_df[ycol]
    model = sm.OLS(y, X, missing="drop").fit()
    resid = model.resid
    sigma = np.sqrt(np.sum(resid**2) / model.df_resid)
    return model, sigma

def score_df(model, sigma, dfin, ycol):
    X = sm.add_constant(dfin[X_cols], has_constant="add")
    pred = model.predict(X)
    z = (dfin[ycol] - pred) / sigma
    out = dfin[["studnr", "sessie", "cohort_num", "aseg_id"]].copy()
    out[f"{ycol}_obs"] = dfin[ycol].values
    out[f"{ycol}_pred"] = pred.values
    out[f"{ycol}_resid"] = (dfin[ycol] - pred).values
    out[f"{ycol}_z"] = z.values
    return out

all_tables = {}
model_summary_rows = []
z_summary_rows = []

for roi in roi_cols:
    model, sigma = fit_model(train, roi)

    model_summary_rows.append({
        "roi": roi,
        "train_N": int(model.nobs),
        "sigma_resid": float(sigma),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "coef_const": float(model.params.get("const", np.nan)),
        "coef_sex": float(model.params.get("sex_bin", np.nan)),
        "coef_age": float(model.params.get("age_imp", np.nan)),
        "coef_age2": float(model.params.get("age2", np.nan)),
        "coef_icv": float(model.params.get(icv_col, np.nan)),
        "p_sex": float(model.pvalues.get("sex_bin", np.nan)),
        "p_age": float(model.pvalues.get("age_imp", np.nan)),
        "p_age2": float(model.pvalues.get("age2", np.nan)),
        "p_icv": float(model.pvalues.get(icv_col, np.nan)),
    })

    train_scored = score_df(model, sigma, train, roi)
    all_tables[f"train_{roi}"] = train_scored

    for gname, gdf in groups.items():
        scored = score_df(model, sigma, gdf, roi)
        all_tables[f"{gname}_{roi}"] = scored

        zcol = f"{roi}_z"
        z_summary_rows.append({
            "roi": roi,
            "group": gname,
            "N": int(scored[zcol].notna().sum()),
            "z_mean": float(scored[zcol].mean(skipna=True)),
            "z_sd": float(scored[zcol].std(skipna=True)),
        })

    # =========================
    # 6) Plots (no seaborn, no subplots)
    # =========================

    # Observed vs predicted (train)
    plt.figure()
    plt.scatter(train_scored[f"{roi}_pred"], train_scored[f"{roi}_obs"])
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title(f"{roi}: Observed vs Predicted (Train: C1 tp1)")
    mn = np.nanmin([train_scored[f"{roi}_pred"].min(), train_scored[f"{roi}_obs"].min()])
    mx = np.nanmax([train_scored[f"{roi}_pred"].max(), train_scored[f"{roi}_obs"].max()])
    plt.plot([mn, mx], [mn, mx])
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{roi}_train_obs_vs_pred.png", dpi=150)
    plt.close()

    # Residual histogram (train)
    plt.figure()
    plt.hist(train_scored[f"{roi}_resid"].dropna(), bins=30)
    plt.xlabel("Residual (Observed - Predicted)")
    plt.ylabel("Count")
    plt.title(f"{roi}: Train residuals (C1 tp1)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{roi}_train_residual_hist.png", dpi=150)
    plt.close()

    # Z-score boxplot across groups
    order = ["C1_tp2", "C1_tp3", "C2_tp1", "C3_tp1", "C4_tp1"]
    box_data, box_labels = [], []
    for gname in order:
        tab = all_tables.get(f"{gname}_{roi}")
        if tab is None:
            continue
        vals = tab[f"{roi}_z"].dropna().values
        box_data.append(vals)
        box_labels.append(gname)

    plt.figure()
    plt.boxplot(box_data, labels=box_labels, showfliers=False)
    plt.axhline(0)
    plt.ylabel("Normative z-score")
    plt.title(f"{roi}: z-scores by group (trained on C1 tp1)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{roi}_z_boxplot_groups.png", dpi=150)
    plt.close()

# =========================
# 7) Save outputs
# =========================
model_summary = pd.DataFrame(model_summary_rows)
z_summary = pd.DataFrame(z_summary_rows)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    model_summary.to_excel(writer, sheet_name="model_summary", index=False)
    z_summary.to_excel(writer, sheet_name="z_summary", index=False)
    for name, tab in all_tables.items():
        tab.to_excel(writer, sheet_name=sanitize_sheet(name), index=False)

print("\nSaved Excel:", OUT_XLSX)
print("Saved plots to:", PLOT_DIR)

# =========================
# 8) ANCOVA: cohort/session effects adjusted for covariates
# =========================
ANCOVA_OUT_SHEET = "ancova"
ANCOVA_TUKEY_SHEET_PREFIX = "tukey_adj_"
ANCOVA_ADJCOL_PREFIX = "adj_"

# Factors + covariates
factor_cols = ["cohort_num", "sessie"]
covar_cols = ["sex_bin", "age_imp", "age2", icv_col]

# Ensure types
df_anc = df.copy()
df_anc["cohort_num"] = df_anc["cohort_num"].astype(int)
df_anc["sessie"] = df_anc["sessie"].astype(str)
for c in covar_cols:
    df_anc[c] = pd.to_numeric(df_anc[c], errors="coerce")

# Keep only rows with everything needed
df_anc = df_anc.dropna(subset=factor_cols + covar_cols + roi_cols).copy()

# Helper: check if cohort×session interaction is estimable (no empty cells)
def full_factorial_ok(d):
    ct = pd.crosstab(d["cohort_num"], d["sessie"])
    return (ct.values > 0).all()

# Helper: build ANCOVA formula
def build_formula(y, d):
    base = f'Q("{y}") ~ C(cohort_num) + C(sessie)'
    # only add interaction if all cohort×sessie cells exist
    if d["cohort_num"].nunique() >= 2 and d["sessie"].nunique() >= 2 and full_factorial_ok(d):
        base = f'Q("{y}") ~ C(cohort_num) * C(sessie)'  # includes main + interaction

    # add covariates (numeric)
    for c in covar_cols:
        base += f' + Q("{c}")'
    return base

# Residualize out covariates (and keep factor effects for downstream Tukey on adjusted values)
# Here we compute "covariate-adjusted ROI" as residuals after removing covariate effects,
# while keeping overall intercept (so the scale looks like volume).
def covariate_adjust(y, d):
    cov_formula = f'Q("{y}") ~ ' + " + ".join([f'Q("{c}")' for c in covar_cols])
    cov_model = smf.ols(cov_formula, data=d).fit()
    # adjusted = residual + grand mean (or intercept); both are fine for group comparisons
    y_adj = cov_model.resid + d[y].mean()
    return y_adj, cov_model

ancova_rows = []
tukey_tables = {}

for roi in roi_cols:
    d = df_anc.dropna(subset=[roi]).copy()

    # ---- ANCOVA on raw ROI with covariates ----
    formula = build_formula(roi, d)
    if PRINT_FORMULA:
        print("\nANCOVA formula:", formula)

    model = smf.ols(formula, data=d).fit()
    aov = anova_lm(model, typ=2)  # Type II SS is a good default for ANCOVA

    # store a nice table with ROI label
    aov_out = aov.reset_index().rename(columns={"index": "term"})
    aov_out.insert(0, "roi", roi)
    tukey_key = f"{roi}_tukey"

    # pull p-values for cohort/session (and interaction if present)
    def get_p(term):
        return float(aov.loc[term, "PR(>F)"]) if term in aov.index else np.nan

    ancova_rows.append({
        "roi": roi,
        "model": "ANCOVA_OLS",
        "formula": formula,
        "p_cohort": get_p("C(cohort_num)"),
        "p_sessie": get_p("C(sessie)"),
        "p_interaction": get_p("C(cohort_num):C(sessie)"),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    })

    all_tables[f"{ANCOVA_OUT_SHEET}_{roi}"] = aov_out

    # ---- Tukey on covariate-adjusted volumes (optional but useful) ----
    # This gives pairwise group differences after "removing" sex/age/age2/ICV.
    d[f"{ANCOVA_ADJCOL_PREFIX}{roi}"], cov_model = covariate_adjust(roi, d)

    # group label as cohort×session for pairwise comparisons
    d["cohort_sessie"] = d["cohort_num"].astype(str) + "_" + d["sessie"].astype(str)

    # Only do Tukey if >=2 groups have data
    if d["cohort_sessie"].nunique() >= 2:
        tuk = pairwise_tukeyhsd(
            endog=d[f"{ANCOVA_ADJCOL_PREFIX}{roi}"].values,
            groups=d["cohort_sessie"].values,
            alpha=ALPHA
        )
        tuk_df = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
        tuk_df.insert(0, "roi", roi)
        tukey_tables[f"{ANCOVA_TUKEY_SHEET_PREFIX}{roi}"] = tuk_df
        all_tables[f"{ANCOVA_TUKEY_SHEET_PREFIX}{roi}"] = tuk_df

    # ---- (Recommended) Mixed model for repeated measures (random intercept by subject) ----
    # This is more correct if the same 'studnr' appears across multiple sessions.
    # It might fail to converge depending on data; wrap in try.
    try:
        mm_formula = formula.replace(f'Q("{roi}") ~ ', f'Q("{roi}") ~ ')
        mixed = smf.mixedlm(mm_formula, data=d, groups=d["studnr"]).fit(reml=False, method="lbfgs", disp=False)
        ancova_rows.append({
            "roi": roi,
            "model": "MixedLM_(1|studnr)",
            "formula": mm_formula,
            "p_cohort": np.nan,     # MixedLM p-values for terms aren’t in an anova table by default
            "p_sessie": np.nan,
            "p_interaction": np.nan,
            "n": int(mixed.nobs),
            "r2": np.nan,
            "adj_r2": np.nan,
        })
        # Save mixed model params table
        mm_tab = mixed.summary().tables[1]
        mm_df = pd.DataFrame(mm_tab.data[1:], columns=mm_tab.data[0])
        mm_df.insert(0, "roi", roi)
        all_tables[f"mixedlm_{roi}"] = mm_df
    except Exception as e:
        print(f"[WARN] MixedLM failed for {roi}: {e}")

# Summary sheet
ancova_summary = pd.DataFrame(ancova_rows)
all_tables["ancova_summary"] = ancova_summary

print("\nANCOVA summary:")
print(ancova_summary)

# --- SAVE ANCOVA SUMMARY SEPARATELY (immediate, explicit paths) ---
ANCOVA_SUMMARY_XLSX = Path("/data/projects/CSC/data/XTC/next_xtc/Normative_modeling_output/ancova_summary.xlsx")
ANCOVA_SUMMARY_CSV  = Path("/data/projects/CSC/data/XTC/next_xtc/Normative_modeling_output/ancova_summary.csv")

# make sure directory exists
ANCOVA_SUMMARY_XLSX.parent.mkdir(parents=True, exist_ok=True)

ancova_summary.to_excel(ANCOVA_SUMMARY_XLSX, index=False)
ancova_summary.to_csv(ANCOVA_SUMMARY_CSV, index=False)

print("Saved ANCOVA summary XLSX ->", ANCOVA_SUMMARY_XLSX.resolve())
print("Saved ANCOVA summary CSV  ->", ANCOVA_SUMMARY_CSV.resolve())
print("File exists (xlsx)?", ANCOVA_SUMMARY_XLSX.exists())
print("File exists (csv)? ", ANCOVA_SUMMARY_CSV.exists())