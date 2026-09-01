"""Generate Supplementary Table S2 from the deidentified demographic CSV."""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

from config import BEHAVIOR_INPUT, DEMOGRAPHICS_OUT, DEMOGRAPHICS_INPUT
from utils import ensure_dirs

SESSION_NAMES = {1.0: "Initial examination", 3.0: "Follow-up"}
GROUPS = ["XTC users", "XTC-naive"]

SOURCE_VARIABLES = {
    "subject_id": "STUDNR",
    "session": "SESSIE",
    "sex": "SEX",
    "age": "LEEFT",
    "IQ": "IQ",
    "education": "HOPLAF",
    "cumulative_XTC_tablets": "XLTTOT",
    "weeks_since_first_XTC": "XFUWKN",
    "weeks_since_last_XTC": "XLTWKN",
    "alcohol_units_per_week": "ALUPW",
    "tobacco_cigarettes_per_week": "TSIGPW",
    "cannabis_joints_last_year": "CA1JTG",
    "amphetamine_times_last_year": "S1JRTO",
    "cocaine_times_last_year": "CO1JTK",
}

EDUCATION_LABELS = {
    1.0: "Primary education",
    2.0: "Lower vocational/general secondary (LBO/MAVO)",
    3.0: "Lower vocational/general secondary (LBO/MAVO)",
    4.0: "Upper secondary/vocational (MBO/HAVO/VWO)",
    5.0: "Upper secondary/vocational (MBO/HAVO/VWO)",
    6.0: "Upper secondary/vocational (MBO/HAVO/VWO)",
    7.0: "Higher professional/university (HBO/WO)",
    8.0: "Higher professional/university (HBO/WO)",
}


def describe(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "N": 0,
            "mean": math.nan,
            "sd": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "N": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def range_number(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if np.isclose(value, round(value), atol=1e-9):
        return str(int(round(value)))
    return f"{value:.1f}"


def formatted_summary(series: pd.Series) -> str:
    stats = describe(series)
    if stats["N"] == 0:
        return "NA"
    return (
        f"{stats['mean']:.1f} ± {stats['sd']:.1f} "
        f"({stats['median']:.1f}; {range_number(stats['min'])}–{range_number(stats['max'])})"
    )


def education_category(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    return EDUCATION_LABELS.get(float(value), np.nan)


def markdown_table(table):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, table.columns)) + " |",
             "| " + " | ".join(["---"] * len(table.columns)) + " |"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in table.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    ensure_dirs([DEMOGRAPHICS_OUT])
    participant = pd.read_csv(DEMOGRAPHICS_INPUT)
    canonical = pd.read_csv(BEHAVIOR_INPUT)[["studnr", "xtc_group"]]
    for frame, key in [(participant, "subject_id"), (canonical, "studnr")]:
        frame[key] = frame[key].astype(str).str.strip().str.upper()
        if frame[key].duplicated().any() or len(frame) != 95:
            raise ValueError("Demographic inputs require 95 unique participants")
    if set(participant.subject_id) != set(canonical.studnr):
        raise ValueError("Demographic participants do not match behavioral input")
    participant = participant.set_index("subject_id").loc[canonical.studnr].reset_index()
    if not (participant.xtc_group.to_numpy() == canonical.xtc_group.to_numpy()).all():
        raise ValueError("Demographic group labels disagree with behavioral input")
    sessions = []
    source_columns = {
        "LEEFT": "age", "HOPLAF": "education_code",
        "ALUPW": "alcohol_units_per_week", "TSIGPW": "tobacco_cigarettes_per_week",
        "CA1JTG": "cannabis_joints_last_year", "S1JRTO": "amphetamine_times_last_year",
        "CO1JTK": "cocaine_times_last_year",
    }
    for session, suffix in [(1.0, "baseline"), (3.0, "followup")]:
        frame = pd.DataFrame({"STUDNR": participant.subject_id, "SESSIE": session,
                              "xtc_group": participant.xtc_group})
        for name, base in source_columns.items():
            frame[name] = participant[f"{base}_{suffix}"]
        frame["SEX"] = participant["sex_baseline" if session == 1 else "sex_followup_recorded"]
        frame["IQ"] = participant.IQ_baseline if session == 1 else np.nan
        # Everyone was XTC-naive at baseline; post-exposure timing is only available at follow-up.
        frame["XLTTOT"] = 0.0 if session == 1 else participant.XTC_cumulative_tablets_followup
        frame["XFUWKN"] = np.nan if session == 1 else participant.weeks_since_first_XTC_followup
        frame["XLTWKN"] = np.nan if session == 1 else participant.weeks_since_last_XTC_followup
        sessions.append(frame)
    selected = pd.concat(sessions, ignore_index=True)

    selected["education_category"] = selected["HOPLAF"].apply(education_category)
    selected["duration_XTC_use_weeks"] = selected["XFUWKN"] - selected["XLTWKN"]

    baseline = selected.loc[selected["SESSIE"] == 1.0].set_index("STUDNR")
    followup = selected.loc[selected["SESSIE"] == 3.0].set_index("STUDNR")

    participant_rows = []
    for subject_id in canonical["studnr"]:
        pre = baseline.loc[subject_id]
        post = followup.loc[subject_id]
        participant_rows.append(
            {
                "subject_id": subject_id,
                "xtc_group": pre["xtc_group"],
                "sex_baseline": pre["SEX"],
                "sex_followup_recorded": post["SEX"],
                "age_baseline": pre["LEEFT"],
                "age_followup": post["LEEFT"],
                "IQ_baseline": pre["IQ"],
                "education_code_baseline": pre["HOPLAF"],
                "education_baseline": pre["education_category"],
                "education_code_followup": post["HOPLAF"],
                "education_followup": post["education_category"],
                "XTC_cumulative_tablets_followup": post["XLTTOT"],
                "weeks_since_first_XTC_followup": post["XFUWKN"],
                "weeks_since_last_XTC_followup": post["XLTWKN"],
                "duration_XTC_use_weeks_followup": post["duration_XTC_use_weeks"],
                "alcohol_units_per_week_baseline": pre["ALUPW"],
                "alcohol_units_per_week_followup": post["ALUPW"],
                "tobacco_cigarettes_per_week_baseline": pre["TSIGPW"],
                "tobacco_cigarettes_per_week_followup": post["TSIGPW"],
                "cannabis_joints_last_year_baseline": pre["CA1JTG"],
                "cannabis_joints_last_year_followup": post["CA1JTG"],
                "amphetamine_times_last_year_baseline": pre["S1JRTO"],
                "amphetamine_times_last_year_followup": post["S1JRTO"],
                "cocaine_times_last_year_baseline": pre["CO1JTK"],
                "cocaine_times_last_year_followup": post["CO1JTK"],
            }
        )
    participant = pd.DataFrame(participant_rows)
    participant.to_csv(
        DEMOGRAPHICS_OUT / "participant_level_demographics_n95_deidentified.csv",
        index=False,
    )

    columns = [
        "Characteristic",
        "XTC users: Initial examination",
        "XTC users: Follow-up",
        "XTC-naive: Initial examination",
        "XTC-naive: Follow-up",
    ]
    table_rows: list[dict[str, str]] = []

    def add_row(characteristic: str, values: list[str]) -> None:
        table_rows.append(dict(zip(columns, [characteristic, *values])))

    user_pre = selected[(selected["xtc_group"] == "XTC users") & (selected["SESSIE"] == 1.0)]
    user_post = selected[(selected["xtc_group"] == "XTC users") & (selected["SESSIE"] == 3.0)]
    naive_pre = selected[(selected["xtc_group"] == "XTC-naive") & (selected["SESSIE"] == 1.0)]
    naive_post = selected[(selected["xtc_group"] == "XTC-naive") & (selected["SESSIE"] == 3.0)]

    def sex_text(frame: pd.DataFrame) -> str:
        male = int((frame["SEX"] == 1).sum())
        female = int((frame["SEX"] == 2).sum())
        return f"{male}/{female}"

    add_row("Sex, male/female, n", [sex_text(user_pre), "NA", sex_text(naive_pre), "NA"])
    add_row(
        "Age, mean ± SD (median; range), y",
        [
            formatted_summary(user_pre["LEEFT"]),
            formatted_summary(user_post["LEEFT"]),
            formatted_summary(naive_pre["LEEFT"]),
            formatted_summary(naive_post["LEEFT"]),
        ],
    )
    add_row(
        "DART-IQ, mean ± SD (median; range)",
        [formatted_summary(user_pre["IQ"]), "NA", formatted_summary(naive_pre["IQ"]), "NA"],
    )
    add_row("Education level, n", ["", "", "", ""])
    for category in [
        "Primary education",
        "Lower vocational/general secondary (LBO/MAVO)",
        "Upper secondary/vocational (MBO/HAVO/VWO)",
        "Higher professional/university (HBO/WO)",
    ]:
        add_row(
            f"  {category}",
            [
                str(int((user_pre["education_category"] == category).sum())),
                str(int((user_post["education_category"] == category).sum())),
                str(int((naive_pre["education_category"] == category).sum())),
                str(int((naive_post["education_category"] == category).sum())),
            ],
        )

    add_row("XTC use, mean ± SD (median; range)", ["", "", "", ""])
    add_row(
        "  Cumulative dose, tablets",
        ["NA", formatted_summary(user_post["XLTTOT"]), "NA", "NA"],
    )
    add_row(
        "  Time since first tablet, wk",
        ["NA", formatted_summary(user_post["XFUWKN"]), "NA", "NA"],
    )
    add_row(
        "  Time since last tablet, wk",
        ["NA", formatted_summary(user_post["XLTWKN"]), "NA", "NA"],
    )
    add_row(
        "  Duration of XTC use, wk",
        ["NA", formatted_summary(user_post["duration_XTC_use_weeks"]), "NA", "NA"],
    )

    add_row("Other substances used in last year", ["", "", "", ""])
    for characteristic, variable in [
        ("  Alcohol, units/wk", "ALUPW"),
        ("  Tobacco, cigarettes/wk", "TSIGPW"),
        ("  Cannabis, joints in last year", "CA1JTG"),
        ("  Amphetamine, times used in last year", "S1JRTO"),
        ("  Cocaine, times used in last year", "CO1JTK"),
    ]:
        add_row(
            characteristic,
            [
                formatted_summary(user_pre[variable]),
                formatted_summary(user_post[variable]),
                formatted_summary(naive_pre[variable]),
                formatted_summary(naive_post[variable]),
            ],
        )

    table = pd.DataFrame(table_rows, columns=columns)
    table.to_csv(
        DEMOGRAPHICS_OUT / "Supplementary_Table_S2_full_demographics.csv",
        index=False,
    )

    markdown = [
        "# Supplementary Table S2. Demographic Characteristics and Use of XTC and Other Substances in the Final Imaging Sample",
        "",
        markdown_table(table),
        "",
        "**Notes.** Values are mean ± standard deviation (median; range) unless otherwise indicated. "
        "XTC users reported incident XTC use between the initial and follow-up examinations; "
        "XTC-naive participants reported no XTC use during this period. Sex was taken from the "
        "initial examination because one participant had a discordant sex code at follow-up. "
        "Follow-up education was missing for one XTC user and one XTC-naive participant. Duration "
        "of XTC use was calculated as weeks since first use minus weeks since last use. Measures of "
        "alcohol, tobacco, cannabis, amphetamine, and cocaine use refer to use during the preceding year.",
    ]
    (DEMOGRAPHICS_OUT / "Supplementary_Table_S2_full_demographics.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )

    summary_rows = []
    for group in GROUPS:
        for session, session_name in SESSION_NAMES.items():
            frame = selected[(selected["xtc_group"] == group) & (selected["SESSIE"] == session)]
            for label, variable in [
                ("Age", "LEEFT"),
                ("IQ", "IQ"),
                ("Cumulative XTC dose", "XLTTOT"),
                ("Weeks since first XTC", "XFUWKN"),
                ("Weeks since last XTC", "XLTWKN"),
                ("Duration XTC use, weeks", "duration_XTC_use_weeks"),
                ("Alcohol units/week", "ALUPW"),
                ("Tobacco cigarettes/week", "TSIGPW"),
                ("Cannabis joints/year", "CA1JTG"),
                ("Amphetamine times/year", "S1JRTO"),
                ("Cocaine times/year", "CO1JTK"),
            ]:
                summary_rows.append(
                    {
                        "group": group,
                        "session": session_name,
                        "measure": label,
                        "source_variable": variable,
                        **describe(frame[variable]),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(
        DEMOGRAPHICS_OUT / "demographic_summary_statistics_long.csv",
        index=False,
    )

    source_map = pd.DataFrame(
        [
            {"table_measure": key, "merged_all_variable": value}
            for key, value in SOURCE_VARIABLES.items()
        ]
        + [
            {
                "table_measure": "duration_XTC_use_weeks",
                "merged_all_variable": "XFUWKN - XLTWKN",
            }
        ]
    )
    source_map.to_csv(
        DEMOGRAPHICS_OUT / "demographic_source_variable_map.csv",
        index=False,
    )

    sex_discordance = participant.loc[
        participant["sex_baseline"] != participant["sex_followup_recorded"],
        ["subject_id", "sex_baseline", "sex_followup_recorded"],
    ]
    sex_discordance.to_csv(
        DEMOGRAPHICS_OUT / "sex_code_discordance_check.csv",
        index=False,
    )

    print("Full demographic table generated from deidentified CSV.")
    print(f"Subjects: {len(participant)}; baseline rows: {len(baseline)}; follow-up rows: {len(followup)}")
    print(f"Sex-code discordances: {len(sex_discordance)}")


if __name__ == "__main__":
    main()
