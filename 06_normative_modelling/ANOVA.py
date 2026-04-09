#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANOVA on hippocampal volumes across cohorts and sessions + boxplots with Tukey brackets.

Fixes:
1) Matplotlib compatibility: uses tick_labels if available, else falls back to labels.
2) Rank-deficient ANOVA: only fits cohort*session interaction if all cohort×session cells exist;
   otherwise falls back to additive model (cohort + session).

Notes:
- This is still a *between-rows* ANOVA. If you have repeated measures per subject across sessions,
  consider a mixed model (random intercept per subject). See MIXED_MODEL flag below.
"""

import os
import re
import numpy as np
import pandas as pd

# ---- plotting (save to file; avoids PyCharm backend crash) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# =========================
# EDIT
# =========================
CSV_PATH = "/data/projects/CSC/data/XTC/next_xtc/next_xtc/aseg_stats.csv"
OUTDIR = "/data/projects/CSC/code/XTC/06_normative_modelling/anova_plots"
ALPHA = 0.05

# If you want x-axis = session only, set to "session"
# If you want x-axis = cohort×session groups (C1TP1...), set to "cohort_session"
PLOT_MODE = "cohort_session"

# If True, fit mixed model (random intercept per subject) instead of ANOVA.
# (No Tukey table for mixed model here; plot/Tukey still uses raw groups.)
MIXED_MODEL = False


# -------------------------
# Column auto-detection
# -------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()

def find_col(df, candidates, contains=False):
    cols = list(df.columns)
    ncols = [norm(c) for c in cols]

    for cand in candidates:
        c = norm(cand)
        if c in ncols:
            return cols[ncols.index(c)]

    if contains:
        for cand in candidates:
            c = norm(cand)
            for i, nc in enumerate(ncols):
                if c in nc:
                    return cols[i]
    return None


df = pd.read_csv(CSV_PATH)
print("Columns:", df.columns.tolist())

COL_ID = find_col(df, ["Measure:volume", "Measure", "ID", "Subject"], contains=True)
COL_COHORT = find_col(df, ["Cohort", "Cohort(1-4)", "Group"], contains=True)
COL_SESS = find_col(df, ["Session", "Session(1-3)"], contains=True)
COL_SESS_LONG = find_col(df, ["Session_long", "Session_long(1-3)"], contains=True)

LEFT_COL = find_col(df, ["Left-Hippocampus", "Left Hippocampus"], contains=True)
RIGHT_COL = find_col(df, ["Right-Hippocampus", "Right Hippocampus"], contains=True)

print("Detected:",
      "\n  ID:", COL_ID,
      "\n  Cohort:", COL_COHORT,
      "\n  Session:", COL_SESS,
      "\n  Session_long:", COL_SESS_LONG,
      "\n  Left:", LEFT_COL,
      "\n  Right:", RIGHT_COL)

for req in [COL_ID, COL_COHORT, COL_SESS, LEFT_COL, RIGHT_COL]:
    if req is None:
        raise ValueError("Missing required column detection. Please print columns and set names manually.")

MEASURES = [LEFT_COL, RIGHT_COL]


# -------------------------
# Helpers
# -------------------------
def parse_subject_id(measure_id: str) -> str:
    m = re.match(r"^(I\d+)", str(measure_id))
    return m.group(1) if m else str(measure_id)

def is_long(measure_id: str) -> bool:
    return ".long." in str(measure_id)

def is_base(measure_id: str) -> bool:
    """
    True only for template/base rows like I010_base.
    Must NOT catch timepoints like I010_tp1.long.I010_base.
    """
    s = str(measure_id)
    return s.endswith("_base") and ("_tp" not in s) and (".long." not in s)

def tp_number_from_id(measure_id: str):
    m = re.search(r"_tp(\d+)", str(measure_id))
    return int(m.group(1)) if m else None

def decide_session_row(row) -> int:
    if COL_SESS_LONG and pd.notna(row.get(COL_SESS_LONG)) and str(row.get(COL_SESS_LONG)).strip() != "":
        return int(row[COL_SESS_LONG])
    if pd.notna(row.get(COL_SESS)) and str(row.get(COL_SESS)).strip() != "":
        return int(row[COL_SESS])
    tp = tp_number_from_id(row[COL_ID])
    return int(tp) if tp is not None else np.nan

def is_multi_session_subject(df_subj: pd.DataFrame) -> bool:
    sess_vals = set()
    if COL_SESS in df_subj.columns:
        sess_vals.update(df_subj[COL_SESS].dropna().astype(int).tolist())
    if COL_SESS_LONG and COL_SESS_LONG in df_subj.columns:
        sess_vals.update(df_subj[COL_SESS_LONG].dropna().astype(int).tolist())

    for v in df_subj[COL_ID].tolist():
        tp = tp_number_from_id(v)
        if tp is not None:
            sess_vals.add(tp)
    return len(sess_vals) >= 2

def build_group_label(cohort: int, session: int) -> str:
    return f"C{int(cohort)}TP{int(session)}"

def p_to_stars(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "n.s."

def add_sig_bracket(ax, x1, x2, y, h, text, lw=1.5):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], linewidth=lw)
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom")


# -------------------------
# Selection using your rules
# -------------------------
df = df.copy()
df[COL_ID] = df[COL_ID].astype(str)
df["subject"] = df[COL_ID].apply(parse_subject_id)
df["is_long"] = df[COL_ID].apply(is_long)
df["is_base"] = df[COL_ID].apply(is_base)
df["session_use"] = df.apply(decide_session_row, axis=1)

# drop base/template rows only (I010_base)
df_nobase = df[~df["is_base"]].copy()

selected_rows = []
for sid, dsub in df_nobase.groupby("subject", sort=True):
    multi = is_multi_session_subject(dsub)

    for sess in sorted(dsub["session_use"].dropna().unique()):
        dsess = dsub[dsub["session_use"] == sess].copy()

        if not multi:
            # single-session: use session (prefer non-long)
            non_long = dsess[~dsess["is_long"]]
            pick = non_long if len(non_long) else dsess
        else:
            # multi-session: prefer long, else non-long
            long = dsess[dsess["is_long"]]
            pick = long if len(long) else dsess

        pick = pick.sort_values(COL_ID).head(1)
        selected_rows.append(pick)

df_sel = pd.concat(selected_rows, ignore_index=True)
df_sel = df_sel.dropna(subset=[COL_COHORT, "session_use"])
df_sel[COL_COHORT] = df_sel[COL_COHORT].astype(int)
df_sel["session_use"] = df_sel["session_use"].astype(int)
df_sel["group"] = df_sel.apply(lambda r: build_group_label(r[COL_COHORT], r["session_use"]), axis=1)

print("\nSelected rows:", len(df_sel))
print(df_sel[[COL_ID, "subject", COL_COHORT, "session_use", "is_long"]].head(25))

# sanity checks
print("\nCohort counts:\n", df_sel[COL_COHORT].value_counts().sort_index())
print("\nSession counts:\n", df_sel["session_use"].value_counts().sort_index())
print("\nLong vs non-long:\n", df_sel["is_long"].value_counts())


# -------------------------
# ANOVA / Mixed model + plot (saved)
# -------------------------
os.makedirs(OUTDIR, exist_ok=True)

def run_best_anova(d: pd.DataFrame, measure: str):
    """
    Run the most appropriate ANOVA given available factor levels.
    - If cohort has <2 levels => test session only
    - If session has <2 levels => test cohort only
    - If both have >=2 levels:
        * if all cohort×session cells exist -> cohort*session
        * else -> cohort + session (no interaction)
    """
    n_cohort = d[COL_COHORT].nunique()
    n_sess = d["session_use"].nunique()

    full_factorial = True
    if n_cohort >= 2 and n_sess >= 2:
        ct = pd.crosstab(d[COL_COHORT], d["session_use"])
        full_factorial = (ct.values > 0).all()

    if n_cohort >= 2 and n_sess >= 2 and full_factorial:
        formula = f'Q("{measure}") ~ C(Q("{COL_COHORT}")) * C(session_use)'
        label = "cohort * session"
    elif n_cohort >= 2 and n_sess >= 2 and not full_factorial:
        formula = f'Q("{measure}") ~ C(Q("{COL_COHORT}")) + C(session_use)'
        label = "cohort + session (no interaction; missing cells)"
    elif n_cohort >= 2 and n_sess < 2:
        formula = f'Q("{measure}") ~ C(Q("{COL_COHORT}"))'
        label = "cohort only"
    elif n_cohort < 2 and n_sess >= 2:
        formula = f'Q("{measure}") ~ C(session_use)'
        label = "session only"
    else:
        return None, "Not enough factor variation for ANOVA (cohort and session both single-level)."

    model = smf.ols(formula, data=d).fit()
    return anova_lm(model, typ=2), f"ANOVA model: {label}"

def run_mixed_model(d: pd.DataFrame, measure: str):
    """
    Mixed model with random intercept for subject.
    """
    n_cohort = d[COL_COHORT].nunique()
    n_sess = d["session_use"].nunique()

    if n_cohort >= 2 and n_sess >= 2:
        formula = f'Q("{measure}") ~ C(Q("{COL_COHORT}")) * C(session_use)'
        label = "MixedLM: cohort * session + (1|subject)"
    elif n_cohort >= 2 and n_sess < 2:
        formula = f'Q("{measure}") ~ C(Q("{COL_COHORT}"))'
        label = "MixedLM: cohort + (1|subject)"
    elif n_cohort < 2 and n_sess >= 2:
        formula = f'Q("{measure}") ~ C(session_use)'
        label = "MixedLM: session + (1|subject)"
    else:
        return None, "Not enough factor variation for model."

    # Random intercept by subject
    md = smf.mixedlm(formula, data=d, groups=d["subject"])
    res = md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
    return res, label

def do_boxplot(ax, data_by_group, xticklabels):
    """
    Matplotlib compatibility: tick_labels (new) vs labels (old).
    """
    try:
        ax.boxplot(data_by_group, tick_labels=xticklabels, showfliers=False)
    except TypeError:
        ax.boxplot(data_by_group, labels=xticklabels, showfliers=False)

def run_anova_and_plot(measure: str):
    d = df_sel.dropna(subset=[measure]).copy()

    print("\n==============================")
    print(f"{measure}")

    if MIXED_MODEL:
        res, msg = run_mixed_model(d, measure)
        print(msg)
        if res is not None:
            print(res.summary())
    else:
        aov, msg = run_best_anova(d, measure)
        print(msg)
        if aov is not None:
            print(aov)

    # x-axis grouping
    if PLOT_MODE == "session":
        xcol = "session_use"
        order = sorted(d[xcol].unique())
        xticklabels = [f"TP{int(x)}" for x in order]
    else:
        xcol = "group"
        order = sorted(d[xcol].unique(),
                       key=lambda s: (int(s.split("TP")[0][1:]), int(s.split("TP")[1])))
        xticklabels = order

    data_by_group = [d[d[xcol] == g][measure].values for g in order]

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(order)), 6))
    do_boxplot(ax, data_by_group, xticklabels)
    ax.set_title(f"{measure} by {xcol}")
    ax.set_ylabel("Volume")
    ax.tick_params(axis="x", rotation=45, labelsize=10)

    # Tukey brackets (only if >=2 groups)
    # (Still computed on raw group labels. If you use MIXED_MODEL, this is not a mixed-model posthoc.)
    if len(order) >= 2:
        tuk = pairwise_tukeyhsd(endog=d[measure].values,
                                groups=d[xcol].astype(str).values,
                                alpha=ALPHA)
        tuk_df = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
        sig = tuk_df[tuk_df["reject"] == True].copy()

        if len(sig):
            pos = {str(lbl): i + 1 for i, lbl in enumerate(order)}  # 1-indexed
            vals = d[measure].values.astype(float)
            ymax = np.nanmax(vals)
            yspan = np.nanmax(vals) - np.nanmin(vals)
            if not np.isfinite(yspan) or yspan == 0:
                yspan = 1.0
            base_y = ymax + 0.05 * yspan
            step = 0.08 * yspan

            sig["g1pos"] = sig["group1"].map(pos)
            sig["g2pos"] = sig["group2"].map(pos)
            sig["dist"] = (sig["g2pos"] - sig["g1pos"]).abs()
            sig = sig.sort_values(["dist", "p-adj"])

            level = 0
            for _, r in sig.iterrows():
                g1, g2, p = str(r["group1"]), str(r["group2"]), float(r["p-adj"])
                if g1 not in pos or g2 not in pos:
                    continue
                x1, x2 = sorted([pos[g1], pos[g2]])
                y = base_y + level * step
                add_sig_bracket(ax, x1, x2, y, h=0.02 * yspan, text=p_to_stars(p))
                level += 1
                if level > 10:
                    break

            ax.set_ylim(top=base_y + (level + 2) * step)

    plt.tight_layout()
    outpath = os.path.join(OUTDIR, f"{measure.replace(' ', '_')}_{xcol}.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved plot:", outpath)

for m in MEASURES:
    run_anova_and_plot(m)