#!/usr/bin/env python3
"""Re-run the NeXT RAVLT behavioral replication in the final N=95 imaging sample.

Key choices
-----------
1. The final imaging sample is defined by the subject IDs in the supplied
   N=95 radiomics mapping CSV.
2. XTC exposure is calculated from XPILLEN3: exposed = XPILLEN3 > 0.
3. Six XPILLEN3 values that are missing in the SAV file are recovered from
   the matching session-3 raw cumulative-dose field XLTTOT in merged_all.csv.
   The two fields are verified to agree wherever both are available.
4. Cannabis, tobacco, alcohol, amphetamine/speed, and cocaine covariates are
   the updated ln(1 + x) variables already stored in the input SAV file.
5. Immediate and delayed recall are analysed by ANCOVA on change scores.
   Recognition decline is analysed with a binomial GLM with a logit link.

The script writes:
- a self-contained 95-subject input CSV;
- a manuscript-oriented results CSV;
- a full model-coefficient CSV.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


# ---------------------------------------------------------------------------
# SAV reader: use pyreadstat when available; otherwise use a small fallback
# reader sufficient for the supplied standard-compressed SPSS system file.
# ---------------------------------------------------------------------------

def _read_sav_fallback(path: Path) -> pd.DataFrame:
    data = path.read_bytes()
    if data[:4] != b"$FL2":
        raise ValueError(f"{path} is not an SPSS system file.")

    case_size = struct.unpack_from("<i", data, 68)[0]
    compression = struct.unpack_from("<i", data, 72)[0]
    n_cases = struct.unpack_from("<i", data, 80)[0]
    bias = struct.unpack_from("<d", data, 84)[0]
    if compression != 1:
        raise ValueError("Fallback reader only supports standard SAV compression.")

    offset = 176
    slots: list[dict[str, Any]] = []
    while True:
        record_type = struct.unpack_from("<i", data, offset)[0]
        if record_type == 2:
            _, var_type, has_label, n_missing, _, _ = struct.unpack_from(
                "<6i", data, offset
            )
            name = data[offset + 24 : offset + 32].decode("latin-1").rstrip(" \x00")
            offset += 32
            if has_label:
                label_length = struct.unpack_from("<i", data, offset)[0]
                offset += 4 + label_length
                offset += (-label_length) % 4
            if n_missing:
                offset += abs(n_missing) * 8
            slots.append({"name": name, "type": var_type})
        elif record_type == 3:
            label_count = struct.unpack_from("<i", data, offset + 4)[0]
            offset += 8
            for _ in range(label_count):
                offset += 8
                text_length = data[offset]
                offset += 1 + text_length
                offset += (-(1 + text_length)) % 8
        elif record_type == 4:
            variable_count = struct.unpack_from("<i", data, offset + 4)[0]
            offset += 8 + 4 * variable_count
        elif record_type == 6:
            line_count = struct.unpack_from("<i", data, offset + 4)[0]
            offset += 8 + 80 * line_count
        elif record_type == 7:
            _, element_size, element_count = struct.unpack_from(
                "<3i", data, offset + 4
            )
            offset += 16 + element_size * element_count
        elif record_type == 999:
            offset += 8
            break
        else:
            raise ValueError(
                f"Unsupported SAV dictionary record {record_type} at byte {offset}."
            )

    if len(slots) != case_size:
        raise ValueError("SAV dictionary size does not match the case size.")

    total_slots = case_size * n_cases
    tokens: list[tuple[str, Any]] = []
    position = offset
    while len(tokens) < total_slots:
        controls = data[position : position + 8]
        if len(controls) != 8:
            raise EOFError("Unexpected end of SAV data.")
        position += 8
        for control in controls:
            if len(tokens) >= total_slots:
                break
            if control == 0:
                continue
            if control == 252:
                raise EOFError("Premature SAV end marker.")
            if control == 253:
                raw = data[position : position + 8]
                position += 8
                tokens.append(("raw", raw))
            elif control == 254:
                tokens.append(("blank", None))
            elif control == 255:
                tokens.append(("missing", None))
            else:
                tokens.append(("small", control))

    variables: list[dict[str, Any]] = []
    slot = 0
    while slot < len(slots):
        info = slots[slot]
        if info["type"] == -1:
            slot += 1
            continue
        if info["type"] == 0:
            variables.append(
                {"name": info["name"], "type": 0, "slot": slot, "width": 1}
            )
            slot += 1
        else:
            width = (info["type"] + 7) // 8
            variables.append(
                {
                    "name": info["name"],
                    "type": info["type"],
                    "slot": slot,
                    "width": width,
                }
            )
            slot += width

    rows: list[dict[str, Any]] = []
    for case in range(n_cases):
        row: dict[str, Any] = {}
        base = case * case_size
        for variable in variables:
            name = variable["name"].lower()
            if variable["type"] == 0:
                kind, payload = tokens[base + variable["slot"]]
                if kind == "small":
                    value = float(payload) - bias
                elif kind == "raw":
                    value = struct.unpack("<d", payload)[0]
                else:
                    value = np.nan
                row[name] = value
            else:
                chunks: list[bytes] = []
                for index in range(variable["width"]):
                    kind, payload = tokens[base + variable["slot"] + index]
                    if kind == "raw":
                        chunks.append(payload)
                    else:
                        chunks.append(b"        ")
                row[name] = (
                    b"".join(chunks)[: variable["type"]]
                    .decode("latin-1", errors="replace")
                    .rstrip(" \x00")
                )
        rows.append(row)

    return pd.DataFrame(rows)


def read_sav(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat  # type: ignore

        dataframe, _ = pyreadstat.read_sav(str(path), apply_value_formats=False)
        dataframe.columns = [str(column).lower() for column in dataframe.columns]
        return dataframe
    except ModuleNotFoundError:
        return _read_sav_fallback(path)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_input(
    sav_path: Path,
    mapping_path: Path,
    raw_long_path: Path,
    input_output_path: Path,
) -> pd.DataFrame:
    sav = read_sav(sav_path)
    mapping = pd.read_csv(mapping_path)
    raw_long = pd.read_csv(raw_long_path, sep=";", low_memory=False)

    if "subject_id" not in mapping.columns:
        raise ValueError("The mapping CSV must contain a subject_id column.")
    imaging_ids = mapping["subject_id"].astype(str).str.strip()
    if imaging_ids.nunique() != 95:
        raise ValueError("The mapping file does not contain exactly 95 unique subjects.")

    required_sav = {
        "studnr",
        "xpillen3",
        "sex",
        "leeftijd",
        "iq",
        "vwsom",
        "bvwsom",
        "vwrec",
        "bvwrec",
        "vwherk",
        "bvwherk",
        "lca1jt",
        "lsigpw",
        "lalupw",
        "ls1jht",
        "lco1jt",
    }
    missing_sav = required_sav.difference(sav.columns)
    if missing_sav:
        raise ValueError(f"Missing SAV variables: {sorted(missing_sav)}")

    analysis = sav.loc[sav["studnr"].astype(str).isin(imaging_ids)].copy()
    if len(analysis) != 95 or analysis["studnr"].nunique() != 95:
        missing_ids = sorted(set(imaging_ids) - set(analysis["studnr"].astype(str)))
        raise ValueError(
            f"Expected 95 matched SAV rows, found {len(analysis)}; missing IDs: {missing_ids}"
        )

    # XPILLEN3 is the session-3 cumulative number of XTC tablets. Six values
    # are missing in the supplied SAV. Recover them from the raw session-3
    # cumulative dose XLTTOT, which is the source variable used to create it.
    raw_session3 = raw_long.loc[
        (raw_long["cohort"].astype(str) == "1")
        & (raw_long["sessie"].astype(str) == "3")
        & raw_long["studnr"].astype(str).isin(imaging_ids),
        ["studnr", "xlttot"],
    ].copy()
    raw_session3["xpillen3_recovered"] = pd.to_numeric(
        raw_session3["xlttot"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    raw_session3 = raw_session3[["studnr", "xpillen3_recovered"]]
    if len(raw_session3) != 95 or raw_session3["studnr"].nunique() != 95:
        raise ValueError("Raw session-3 data do not provide exactly one row per imaging subject.")
    if raw_session3["xpillen3_recovered"].isna().any():
        raise ValueError("Some raw session-3 XTC doses could not be parsed.")

    analysis = analysis.merge(
        raw_session3, on="studnr", how="left", validate="one_to_one"
    )
    analysis["xpillen3_sav"] = analysis["xpillen3"]

    both = analysis["xpillen3_sav"].notna()
    if not np.allclose(
        analysis.loc[both, "xpillen3_sav"],
        analysis.loc[both, "xpillen3_recovered"],
        atol=1e-12,
        rtol=0,
    ):
        differences = analysis.loc[
            both
            & ~np.isclose(
                analysis["xpillen3_sav"],
                analysis["xpillen3_recovered"],
                atol=1e-12,
                rtol=0,
            ),
            ["studnr", "xpillen3_sav", "xpillen3_recovered"],
        ]
        raise ValueError(f"XPILLEN3 disagrees with raw session-3 dose:\n{differences}")

    analysis["xpillen3_source"] = np.where(
        analysis["xpillen3_sav"].notna(),
        "SAV XPILLEN3",
        "recovered from raw session-3 XLTTOT",
    )
    analysis["xpillen3"] = analysis["xpillen3_sav"].fillna(
        analysis["xpillen3_recovered"]
    )
    analysis["xtc_exposure"] = (analysis["xpillen3"] > 0).astype(int)
    analysis["xtc_group"] = analysis["xtc_exposure"].map(
        {1: "XTC users", 0: "XTC-naive"}
    )
    analysis["log1p_xpillen3"] = np.log1p(analysis["xpillen3"])

    group_counts = analysis["xtc_exposure"].value_counts().to_dict()
    if group_counts != {0: 48, 1: 47}:
        raise ValueError(f"Expected 47 XTC users and 48 XTC-naive subjects; found {group_counts}")

    analysis["ravlt_immediate_change"] = analysis["bvwsom"] - analysis["vwsom"]
    analysis["ravlt_delayed_change"] = analysis["bvwrec"] - analysis["vwrec"]
    analysis["ravlt_recognition_change"] = analysis["bvwherk"] - analysis["vwherk"]
    analysis["ravlt_recognition_decline"] = (
        analysis["ravlt_recognition_change"] < 0
    ).astype(int)

    # Add readable aliases while retaining original variable names.
    analysis["age"] = analysis["leeftijd"]
    analysis["dart_iq"] = analysis["iq"]
    analysis["log1p_cannabis"] = analysis["lca1jt"]
    analysis["log1p_tobacco"] = analysis["lsigpw"]
    analysis["log1p_alcohol"] = analysis["lalupw"]
    analysis["log1p_amphetamine"] = analysis["ls1jht"]
    analysis["log1p_cocaine"] = analysis["lco1jt"]

    output_columns = [
        "studnr",
        "xtc_group",
        "xtc_exposure",
        "xpillen3",
        "log1p_xpillen3",
        "xpillen3_sav",
        "xpillen3_recovered",
        "xpillen3_source",
        "sex",
        "age",
        "dart_iq",
        "vwsom",
        "bvwsom",
        "ravlt_immediate_change",
        "vwrec",
        "bvwrec",
        "ravlt_delayed_change",
        "vwherk",
        "bvwherk",
        "ravlt_recognition_change",
        "ravlt_recognition_decline",
        "log1p_cannabis",
        "log1p_tobacco",
        "log1p_alcohol",
        "log1p_amphetamine",
        "log1p_cocaine",
    ]
    prepared = analysis[output_columns].sort_values("studnr").reset_index(drop=True)
    prepared.to_csv(input_output_path, index=False)
    return prepared


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def partial_eta_squared(anova_table: pd.DataFrame, term: str) -> float:
    return float(
        anova_table.loc[term, "sum_sq"]
        / (anova_table.loc[term, "sum_sq"] + anova_table.loc["Residual", "sum_sq"])
    )


def describe_group(
    data: pd.DataFrame,
    group: int,
    baseline: str,
    followup: str,
    change: str,
) -> dict[str, float]:
    subset = data.loc[data["xtc_exposure"] == group]
    return {
        "n": int(len(subset)),
        "baseline_mean": float(subset[baseline].mean()),
        "baseline_sd": float(subset[baseline].std(ddof=1)),
        "followup_mean": float(subset[followup].mean()),
        "followup_sd": float(subset[followup].std(ddof=1)),
        "change_mean": float(subset[change].mean()),
        "change_sd": float(subset[change].std(ddof=1)),
    }


def run_analysis(
    input_csv: Path,
    results_output_path: Path,
    coefficients_output_path: Path,
) -> pd.DataFrame:
    data = pd.read_csv(input_csv)
    if len(data) != 95:
        raise ValueError("The prepared input CSV must contain exactly 95 rows.")

    covariates = [
        "age",
        "dart_iq",
        "log1p_cannabis",
        "log1p_tobacco",
        "log1p_alcohol",
        "log1p_amphetamine",
        "log1p_cocaine",
    ]

    published = {
        "RAVLT immediate recall, words": {
            "published_sample": "58 XTC users; 60 matched controls",
            "published_statistic": "F = 4.62",
            "published_p": 0.030,
        },
        "RAVLT delayed recall, words": {
            "published_sample": "58 XTC users; 60 matched controls",
            "published_statistic": "F = 4.67",
            "published_p": 0.030,
        },
        "RAVLT recognition, words": {
            "published_sample": "58 XTC users; 60 matched controls",
            "published_statistic": "OR = 5.87",
            "published_p": 0.020,
        },
    }

    rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    continuous_outcomes = [
        (
            "RAVLT immediate recall, words",
            "vwsom",
            "bvwsom",
            "ravlt_immediate_change",
        ),
        (
            "RAVLT delayed recall, words",
            "vwrec",
            "bvwrec",
            "ravlt_delayed_change",
        ),
    ]

    for outcome, baseline, followup, change in continuous_outcomes:
        formula = (
            f"{change} ~ C(xtc_exposure) + C(sex) + {baseline} + "
            + " + ".join(covariates)
        )
        model_data = data[
            [change, "xtc_exposure", "sex", baseline] + covariates
        ].dropna()
        model = smf.ols(formula, data=model_data).fit()
        anova3 = anova_lm(model, typ=3)
        anova_term = "C(xtc_exposure)"
        coefficient_term = "C(xtc_exposure)[T.1]"
        confidence_interval = model.conf_int().loc[coefficient_term]

        users = describe_group(data, 1, baseline, followup, change)
        naive = describe_group(data, 0, baseline, followup, change)
        p_value = float(anova3.loc[anova_term, "PR(>F)"])

        rows.append(
            {
                "outcome": outcome,
                "present_sample_n": len(model_data),
                "present_n_xtc_users": int((model_data["xtc_exposure"] == 1).sum()),
                "present_n_xtc_naive": int((model_data["xtc_exposure"] == 0).sum()),
                "xtc_users_baseline_mean": users["baseline_mean"],
                "xtc_users_baseline_sd": users["baseline_sd"],
                "xtc_users_followup_mean": users["followup_mean"],
                "xtc_users_followup_sd": users["followup_sd"],
                "xtc_users_change_mean": users["change_mean"],
                "xtc_users_change_sd": users["change_sd"],
                "xtc_users_decline_n": np.nan,
                "xtc_users_decline_pct": np.nan,
                "xtc_naive_baseline_mean": naive["baseline_mean"],
                "xtc_naive_baseline_sd": naive["baseline_sd"],
                "xtc_naive_followup_mean": naive["followup_mean"],
                "xtc_naive_followup_sd": naive["followup_sd"],
                "xtc_naive_change_mean": naive["change_mean"],
                "xtc_naive_change_sd": naive["change_sd"],
                "xtc_naive_decline_n": np.nan,
                "xtc_naive_decline_pct": np.nan,
                "model": "ANCOVA on change score",
                "adjusted_effect": float(model.params[coefficient_term]),
                "adjusted_effect_type": "B, XTC users minus XTC-naive",
                "ci_low": float(confidence_interval.iloc[0]),
                "ci_high": float(confidence_interval.iloc[1]),
                "test_statistic": float(anova3.loc[anova_term, "F"]),
                "test_statistic_type": "F",
                "p": p_value,
                "partial_eta2": partial_eta_squared(anova3, anova_term),
                "odds_ratio": np.nan,
                "published_sample": published[outcome]["published_sample"],
                "published_statistic": published[outcome]["published_statistic"],
                "published_p": published[outcome]["published_p"],
                "interpretation": (
                    "Replicated" if p_value < 0.05 else "Not statistically replicated"
                ),
                "analysis_note": "All five polysubstance covariates use ln(1 + x).",
            }
        )

        for term in model.params.index:
            ci = model.conf_int().loc[term]
            coefficient_rows.append(
                {
                    "outcome": outcome,
                    "model": "OLS ANCOVA",
                    "term": term,
                    "coefficient": float(model.params[term]),
                    "standard_error": float(model.bse[term]),
                    "ci_low": float(ci.iloc[0]),
                    "ci_high": float(ci.iloc[1]),
                    "test_statistic": float(model.tvalues[term]),
                    "test_statistic_type": "t",
                    "p": float(model.pvalues[term]),
                    "odds_ratio": np.nan,
                }
            )

    # Recognition: use a binomial GLM with logit link. This is the same logistic
    # model as smf.logit, but IRLS supplies stable XTC inference despite sparse
    # amphetamine exposure causing a singular/non-converged discrete Logit Hessian.
    outcome = "RAVLT recognition, words"
    formula = (
        "ravlt_recognition_decline ~ xtc_exposure + sex + vwherk + "
        + " + ".join(covariates)
    )
    recognition_data = data[
        ["ravlt_recognition_decline", "xtc_exposure", "sex", "vwherk"]
        + covariates
    ].dropna()
    recognition_model = smf.glm(
        formula,
        data=recognition_data,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
    ).fit()

    term = "xtc_exposure"
    log_or_ci = recognition_model.conf_int().loc[term]
    users = describe_group(
        data,
        1,
        "vwherk",
        "bvwherk",
        "ravlt_recognition_change",
    )
    naive = describe_group(
        data,
        0,
        "vwherk",
        "bvwherk",
        "ravlt_recognition_change",
    )
    user_decline_n = int(
        data.loc[data["xtc_exposure"] == 1, "ravlt_recognition_decline"].sum()
    )
    naive_decline_n = int(
        data.loc[data["xtc_exposure"] == 0, "ravlt_recognition_decline"].sum()
    )
    p_value = float(recognition_model.pvalues[term])

    rows.append(
        {
            "outcome": outcome,
            "present_sample_n": len(recognition_data),
            "present_n_xtc_users": int((recognition_data["xtc_exposure"] == 1).sum()),
            "present_n_xtc_naive": int((recognition_data["xtc_exposure"] == 0).sum()),
            "xtc_users_baseline_mean": users["baseline_mean"],
            "xtc_users_baseline_sd": users["baseline_sd"],
            "xtc_users_followup_mean": users["followup_mean"],
            "xtc_users_followup_sd": users["followup_sd"],
            "xtc_users_change_mean": users["change_mean"],
            "xtc_users_change_sd": users["change_sd"],
            "xtc_users_decline_n": user_decline_n,
            "xtc_users_decline_pct": 100 * user_decline_n / users["n"],
            "xtc_naive_baseline_mean": naive["baseline_mean"],
            "xtc_naive_baseline_sd": naive["baseline_sd"],
            "xtc_naive_followup_mean": naive["followup_mean"],
            "xtc_naive_followup_sd": naive["followup_sd"],
            "xtc_naive_change_mean": naive["change_mean"],
            "xtc_naive_change_sd": naive["change_sd"],
            "xtc_naive_decline_n": naive_decline_n,
            "xtc_naive_decline_pct": 100 * naive_decline_n / naive["n"],
            "model": "Binomial GLM with logit link",
            "adjusted_effect": float(recognition_model.params[term]),
            "adjusted_effect_type": "log OR, XTC users versus XTC-naive",
            "ci_low": float(np.exp(log_or_ci.iloc[0])),
            "ci_high": float(np.exp(log_or_ci.iloc[1])),
            "test_statistic": float(recognition_model.tvalues[term]),
            "test_statistic_type": "Wald z",
            "p": p_value,
            "partial_eta2": np.nan,
            "odds_ratio": float(np.exp(recognition_model.params[term])),
            "published_sample": published[outcome]["published_sample"],
            "published_statistic": published[outcome]["published_statistic"],
            "published_p": published[outcome]["published_p"],
            "interpretation": (
                "Replicated" if p_value < 0.05 else "Not statistically replicated"
            ),
            "analysis_note": (
                "All five polysubstance covariates use ln(1 + x). "
                "GLM-logit was used because sparse amphetamine exposure made the "
                "discrete Logit Hessian singular/non-convergent."
            ),
        }
    )

    for coefficient_term in recognition_model.params.index:
        ci = recognition_model.conf_int().loc[coefficient_term]
        coefficient_rows.append(
            {
                "outcome": outcome,
                "model": "Binomial GLM-logit",
                "term": coefficient_term,
                "coefficient": float(recognition_model.params[coefficient_term]),
                "standard_error": float(recognition_model.bse[coefficient_term]),
                "ci_low": float(ci.iloc[0]),
                "ci_high": float(ci.iloc[1]),
                "test_statistic": float(recognition_model.tvalues[coefficient_term]),
                "test_statistic_type": "Wald z",
                "p": float(recognition_model.pvalues[coefficient_term]),
                "odds_ratio": float(np.exp(recognition_model.params[coefficient_term])),
            }
        )

    results = pd.DataFrame(rows)
    coefficients = pd.DataFrame(coefficient_rows)
    results.to_csv(results_output_path, index=False)
    coefficients.to_csv(coefficients_output_path, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sav",
        type=Path,
        default=Path("ravlt_plus_baseline_covariates_log1p_poly.sav"),
        help="Updated SAV containing ln(1+x) polysubstance covariates.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(
            "radiomics_hippocampus_thalamus_pre_delta_95_subjects_with_covariates_log1p_poly.csv"
        ),
        help="CSV containing the final 95 imaging subject IDs.",
    )
    parser.add_argument(
        "--raw-long",
        type=Path,
        default=Path("merged_all.csv"),
        help="Original long-format raw CSV used to recover missing XPILLEN3 values.",
    )
    parser.add_argument(
        "--input-out",
        type=Path,
        default=Path("ravlt_n95_log1p_xpillen3_input.csv"),
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=Path("ravlt_n95_behavioral_replication_results_log1p_xpillen3.csv"),
    )
    parser.add_argument(
        "--coefficients-out",
        type=Path,
        default=Path("ravlt_n95_full_model_coefficients_log1p_xpillen3.csv"),
    )
    args = parser.parse_args()

    prepared = prepare_input(
        args.sav,
        args.mapping,
        args.raw_long,
        args.input_out,
    )
    results = run_analysis(
        args.input_out,
        args.results_out,
        args.coefficients_out,
    )

    recovered = int((prepared["xpillen3_source"] != "SAV XPILLEN3").sum())
    print(f"Prepared input: {args.input_out} ({len(prepared)} subjects)")
    print(
        "Groups: "
        f"{int((prepared.xtc_exposure == 1).sum())} XTC users; "
        f"{int((prepared.xtc_exposure == 0).sum())} XTC-naive"
    )
    print(f"Recovered missing XPILLEN3 values: {recovered}")
    print(f"Results: {args.results_out}")
    print(f"Full coefficients: {args.coefficients_out}")
    print("\nKey present-sample results:")
    for _, row in results.iterrows():
        if row["test_statistic_type"] == "F":
            print(
                f"- {row['outcome']}: F={row['test_statistic']:.4f}, "
                f"p={row['p']:.6f}, partial eta2={row['partial_eta2']:.4f}"
            )
        else:
            print(
                f"- {row['outcome']}: OR={row['odds_ratio']:.4f}, "
                f"95% CI={row['ci_low']:.4f}–{row['ci_high']:.4f}, "
                f"p={row['p']:.6f}"
            )


if __name__ == "__main__":
    main()
