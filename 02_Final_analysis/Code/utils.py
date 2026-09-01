"""Shared helpers for the XTC manuscript analysis pipeline."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_numeric(path: Path, id_columns=("subject_id", "studnr")) -> pd.DataFrame:
    """Read a CSV and safely convert decimal-comma numeric strings."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if c in id_columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
        elif df[c].dtype == object:
            converted = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False).str.strip(),
                errors="coerce",
            )
            # Convert only when at least one nonmissing value was numeric.
            if converted.notna().any():
                df[c] = converted
    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def validate_unique_subjects(df: pd.DataFrame, subject_col: str = "subject_id") -> None:
    require_columns(df, [subject_col], "Dataset")
    if df[subject_col].duplicated().any():
        duplicates = df.loc[df[subject_col].duplicated(), subject_col].tolist()
        raise ValueError(f"Duplicate subject IDs: {duplicates[:10]}")


def mean_impute(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(numeric.mean())


def centered(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric - numeric.mean()


def add_group(df: pd.DataFrame, dose_col: str = "xlttot_sessie3") -> pd.DataFrame:
    result = df.copy()
    result[dose_col] = pd.to_numeric(result[dose_col], errors="coerce")
    result["xtc_group"] = np.where(result[dose_col] > 0, "XTC users", "XTC-naive")
    result["xtc_user"] = (result["xtc_group"] == "XTC users").astype(int)
    return result


def fit_mixedlm(formula: str, data: pd.DataFrame, group_col: str):
    """Fit a random-intercept mixed model with controlled fallbacks."""
    import statsmodels.formula.api as smf
    errors = []
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data[group_col],
                    re_formula="1",
                )
                result = model.fit(reml=False, method=method, maxiter=2000, disp=False)
            return result, method
        except Exception as exc:  # pragma: no cover - fallback path depends on platform
            errors.append(f"{method}: {exc!r}")
    raise RuntimeError("MixedLM failed with all optimizers:\n" + "\n".join(errors))


def tidy_model_result(result, model_name: str, extra: dict | None = None) -> pd.DataFrame:
    extra = {} if extra is None else dict(extra)
    conf = result.conf_int()
    rows = []
    for term in result.params.index:
        row = {
            "model": model_name,
            "term": term,
            "beta": float(result.params[term]),
            "std_error": float(result.bse.get(term, np.nan)),
            "statistic": float(result.tvalues.get(term, np.nan)),
            "p": float(result.pvalues.get(term, np.nan)),
            "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
            "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
        }
        row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_fdr(df: pd.DataFrame, p_col: str = "p", output_col: str = "FDR_q") -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests
    result = df.copy()
    result[output_col] = np.nan
    valid = result[p_col].notna()
    if valid.any():
        result.loc[valid, output_col] = multipletests(
            result.loc[valid, p_col].astype(float), method="fdr_bh"
        )[1]
    return result


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
