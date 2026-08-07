import pandas as pd
import numpy as np
import warnings
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

vol_path = "whole_brain_volume_pre_delta(3).csv"
dose_path = "xtc_dose_wide_mapped_to_y(4).csv"

vol = pd.read_csv(vol_path, dtype=str)
for c in vol.columns:
    if c != "subject_id":
        vol[c] = pd.to_numeric(vol[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

dose = pd.read_csv(dose_path, dtype=str)
for c in ["cohort_x", "vwrec", "xlttot_sessie1", "xlttot_sessie2", "xlttot_sessie3"]:
    dose[c] = pd.to_numeric(dose[c], errors="coerce")

vol["subject_id"] = vol["subject_id"].astype(str)
dose["subject_code"] = dose["subject_code"].astype(str)

df = vol.merge(dose, left_on="subject_id", right_on="subject_code", how="left", indicator=True)
sample = df.dropna(subset=["xlttot_sessie3"]).copy()
sample["xtc_group"] = np.where(sample["xlttot_sessie3"] > 0, "XTC users", "XTC-naive")
sample["xtc_user"] = (sample["xtc_group"] == "XTC users").astype(int)
sample["BrainSegVol_pre"] = sample["aseg+DKT_BrainSegVol_pre"]
sample["BrainSegVol_fu"] = sample["aseg+DKT_BrainSegVol_pre"] + sample["aseg+DKT_BrainSegVol_delta"]

rois = {
    "Left hippocampus": "aseg+DKT_Left-Hippocampus",
    "Right hippocampus": "aseg+DKT_Right-Hippocampus",
    "Left thalamus": "aseg+DKT_Left-Thalamus",
    "Right thalamus": "aseg+DKT_Right-Thalamus",
}

def make_long(dat, colbase):
    long = pd.DataFrame({
        "subject_id": np.repeat(dat["subject_id"].values, 2),
        "sex": np.repeat(dat["sex"].values, 2),
        "xtc_group": np.repeat(dat["xtc_group"].values, 2),
        "xtc_user": np.repeat(dat["xtc_user"].values, 2),
        "session": np.tile([0, 1], len(dat)),
        "BrainSegVol": np.ravel(np.column_stack([dat["BrainSegVol_pre"].values, dat["BrainSegVol_fu"].values])),
        "ROI_volume": np.ravel(np.column_stack([dat[colbase + "_pre"].values, dat[colbase + "_pre"].values + dat[colbase + "_delta"].values])),
    }).dropna()
    long["BrainSegVol_k"] = long["BrainSegVol"] / 1000.0
    return long

def fit_roi(dat, roi_name, colbase, interaction=False):
    long = make_long(dat, colbase)
    formula = "ROI_volume ~ session * xtc_user + C(sex) + BrainSegVol_k" if interaction else "ROI_volume ~ session + C(sex) + BrainSegVol_k"
    model = mixedlm(formula, long, groups=long["subject_id"])
    try:
        res = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    except Exception:
        res = model.fit(reml=False, method="powell", maxiter=1000, disp=False)

    term = "session:xtc_user" if interaction else "session"
    est = float(res.params[term])
    se = float(res.bse[term])
    z = est / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "Region": roi_name,
        "N": int(long["subject_id"].nunique()),
        "Estimate, follow-up - baseline": est,
        "SE": se,
        "95% CI lower": est - 1.96 * se,
        "95% CI upper": est + 1.96 * se,
        "z": z,
        "p": p,
        "Converged": bool(res.converged),
    }

rows = []
for group in ["XTC-naive", "XTC users"]:
    group_data = sample[sample["xtc_group"] == group]
    group_rows = [fit_roi(group_data, roi_name, colbase) for roi_name, colbase in rois.items()]
    qvals = multipletests([r["p"] for r in group_rows], method="fdr_bh")[1]
    for r, q in zip(group_rows, qvals):
        r["Group"] = group
        r["FDR q"] = float(q)
        rows.append(r)

stratified = pd.DataFrame(rows)
stratified.to_csv("roi_lme_by_xtc_group_full_results.csv", index=False)

interaction_rows = []
for roi_name, colbase in rois.items():
    r = fit_roi(sample, roi_name, colbase, interaction=True)
    r["Term"] = "session x XTC-user interaction"
    interaction_rows.append(r)

interaction = pd.DataFrame(interaction_rows)
interaction["FDR q"] = multipletests(interaction["p"], method="fdr_bh")[1]
interaction.to_csv("roi_lme_session_by_xtc_interaction.csv", index=False)

print("Final sample:", len(sample))
print(sample["xtc_group"].value_counts())
print(stratified[["Group", "Region", "N", "Estimate, follow-up - baseline", "95% CI lower", "95% CI upper", "z", "p", "FDR q"]])
print(interaction[["Region", "N", "Estimate, follow-up - baseline", "95% CI lower", "95% CI upper", "z", "p", "FDR q"]])
