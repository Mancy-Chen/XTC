"""Bootstrap stability of the baseline-derived whole-brain PC1 solution.

The resampling isolates PCA-axis uncertainty: each bootstrap PCA is fitted to a
resample of baseline participants, sign-aligned to the original PC1, and then
used to project the complete original baseline and follow-up datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    BOOTSTRAP_WHOLE_BRAIN_OUT, GROUP_ORDER, N_BOOTSTRAP, PCA_WHOLE_BRAIN_OUT,
    RANDOM_SEED, VARIANCE_THRESHOLD, WHOLE_BRAIN_INPUT,
)
from utils import add_group, ensure_dirs, read_csv_numeric


def identify_features(df: pd.DataFrame) -> list[str]:
    excluded = {"vwrec_pre", "aseg+DKT_BrainSegVol_pre"}
    return [c for c in df.columns if c.endswith("_pre") and c not in excluded and f"{c[:-4]}_delta" in df.columns]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def residualize(y: np.ndarray, predictors: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(y)), predictors])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def main() -> None:
    ensure_dirs([BOOTSTRAP_WHOLE_BRAIN_OUT])
    raw = read_csv_numeric(WHOLE_BRAIN_INPUT)
    data = add_group(raw)
    pre_cols = identify_features(data)
    variances = data[pre_cols].var(axis=0, ddof=0)
    kept = variances[variances > VARIANCE_THRESHOLD].index.tolist()
    bases = [c[:-4] for c in kept]
    delta_cols = [f"{b}_delta" for b in bases]
    X_pre = data[kept].to_numpy(float)
    X_delta = data[delta_cols].to_numpy(float)
    X_post = X_pre + X_delta

    original_loadings_df = read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_loadings.csv")
    original_loadings_df = original_loadings_df.set_index("feature").loc[bases].reset_index()
    original_loading = original_loadings_df["PC1_loading"].to_numpy(float)
    original_scores = read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_scores_wide.csv")
    original_pre = original_scores["PC1_pre"].to_numpy(float)
    original_delta = original_scores["PC1_delta"].to_numpy(float)
    brainseg = original_scores["aseg+DKT_BrainSegVol_pre"].to_numpy(float)
    logdose = original_scores["log1p_xtc"].to_numpy(float)
    dose_c = logdose - np.mean(logdose)
    memory = original_scores["vwrec_delta"].to_numpy(float)
    groups = add_group(original_scores)["xtc_group"].to_numpy()
    original_top20 = set(np.argsort(np.abs(original_loading))[-20:])

    rng = np.random.default_rng(RANDOM_SEED)
    iteration_rows = []
    loading_matrix = np.empty((N_BOOTSTRAP, len(kept)), dtype=float)

    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(data), size=len(data))
        scaler = StandardScaler().fit(X_pre[idx])
        Z_boot = scaler.transform(X_pre[idx])
        pca = PCA(n_components=1, svd_solver="full").fit(Z_boot)
        loading = pca.components_[0].copy()
        if np.dot(loading, original_loading) < 0:
            loading *= -1
        loading_matrix[b] = loading

        score_pre = scaler.transform(X_pre) @ loading
        score_post = scaler.transform(X_post) @ loading
        score_delta = score_post - score_pre
        adjusted_delta = residualize(score_delta, np.column_stack([score_pre, brainseg]))

        row = {
            "bootstrap_iteration": b + 1,
            "PC1_explained_variance": float(pca.explained_variance_ratio_[0]),
            "loading_cosine_similarity": cosine(original_loading, loading),
            "baseline_PC1_score_correlation": corr(original_pre, score_pre),
            "delta_PC1_correlation": corr(original_delta, score_delta),
            "original_top20_overlap": len(original_top20 & set(np.argsort(np.abs(loading))[-20:])),
            "time_by_dose_coefficient": float(np.linalg.lstsq(np.column_stack([np.ones(len(dose_c)), dose_c]), score_delta, rcond=None)[0][1]),
            "raw_rho_overall": float(spearmanr(score_delta, memory).statistic),
            "adjusted_rho_overall": float(spearmanr(adjusted_delta, memory).statistic),
        }
        for group in GROUP_ORDER:
            mask = groups == group
            suffix = "naive" if group == "XTC-naive" else "users"
            row[f"raw_rho_{suffix}"] = float(spearmanr(score_delta[mask], memory[mask]).statistic)
            row[f"adjusted_rho_{suffix}"] = float(spearmanr(adjusted_delta[mask], memory[mask]).statistic)
        iteration_rows.append(row)
        if (b + 1) % 100 == 0:
            print(f"Bootstrap {b + 1}/{N_BOOTSTRAP}")

    iterations = pd.DataFrame(iteration_rows)
    iterations.to_csv(BOOTSTRAP_WHOLE_BRAIN_OUT / "whole_brain_pca_bootstrap_iterations.csv", index=False)

    original_adjusted = residualize(original_delta, np.column_stack([original_pre, brainseg]))
    original_values = {
        "PC1_explained_variance": float(read_csv_numeric(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_explained_variance.csv").iloc[0]["explained_variance_ratio"]),
        "loading_cosine_similarity": 1.0,
        "baseline_PC1_score_correlation": 1.0,
        "delta_PC1_correlation": 1.0,
        "original_top20_overlap": 20.0,
        "time_by_dose_coefficient": float(np.linalg.lstsq(np.column_stack([np.ones(len(dose_c)), dose_c]), original_delta, rcond=None)[0][1]),
        "raw_rho_overall": float(spearmanr(original_delta, memory).statistic),
        "adjusted_rho_overall": float(spearmanr(original_adjusted, memory).statistic),
    }
    for group in GROUP_ORDER:
        mask = groups == group; suffix = "naive" if group == "XTC-naive" else "users"
        original_values[f"raw_rho_{suffix}"] = float(spearmanr(original_delta[mask], memory[mask]).statistic)
        original_values[f"adjusted_rho_{suffix}"] = float(spearmanr(original_adjusted[mask], memory[mask]).statistic)

    summary_rows = []
    for metric in iterations.columns.drop("bootstrap_iteration"):
        values = iterations[metric].dropna().to_numpy(float)
        summary_rows.append({"measure": metric, "original": original_values.get(metric, np.nan), "bootstrap_median": float(np.median(values)), "percentile_2_5": float(np.percentile(values, 2.5)), "percentile_97_5": float(np.percentile(values, 97.5)), "N_bootstrap": len(values)})
    pd.DataFrame(summary_rows).to_csv(BOOTSTRAP_WHOLE_BRAIN_OUT / "whole_brain_pca_bootstrap_summary.csv", index=False)

    loading_summary = original_loadings_df[["feature", "PC1_loading"]].copy()
    loading_summary["bootstrap_median"] = np.median(loading_matrix, axis=0)
    loading_summary["percentile_2_5"] = np.percentile(loading_matrix, 2.5, axis=0)
    loading_summary["percentile_97_5"] = np.percentile(loading_matrix, 97.5, axis=0)
    original_sign = np.sign(original_loading)
    loading_summary["same_sign_percentage"] = (np.sign(loading_matrix) == original_sign).mean(axis=0) * 100
    loading_summary["original_absolute_rank"] = pd.Series(np.abs(original_loading)).rank(method="min", ascending=False).astype(int).to_numpy()
    loading_summary.to_csv(BOOTSTRAP_WHOLE_BRAIN_OUT / "whole_brain_pc1_loading_bootstrap_summary.csv", index=False)
    print("Whole-brain PCA bootstrap stability completed.")


if __name__ == "__main__":
    main()
