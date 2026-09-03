"""Baseline-fit PCA for predefined hippocampal/thalamic radiomics features."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    N_PCS_EXPORT,
    PCA_PREDEFINED_OUT,
    PREDEFINED_ROI_INPUT,
    PREDEFINED_ROIS,
    VARIANCE_THRESHOLD,
)
from utils import ensure_dirs, read_csv_numeric


def analysis_definitions(df: pd.DataFrame) -> dict[str, list[str]]:
    paired_pre = [
        c for c in df.columns
        if c.endswith("_pre")
        and c not in {"vwrec_pre", "aseg+DKT_BrainSegVol_pre"}
        and f"{c[:-4]}_delta" in df.columns
    ]
    definitions = {
        "Combined_shape_only": [c for c in paired_pre if "_shape_" in c],
        "Combined_shape_firstorder": [
            c for c in paired_pre if "_shape_" in c or "_firstorder_" in c
        ],
    }
    return definitions


def main() -> None:
    ensure_dirs([PCA_PREDEFINED_OUT])
    df = read_csv_numeric(PREDEFINED_ROI_INPUT)
    definitions = analysis_definitions(df)

    metadata_cols = [
        "subject_id", "sex", "age", "xlttot_sessie3", "log1p_xtc", "iq",
        "vwrec_pre", "vwrec_delta", "lca1jt", "lsigpw", "lalupw", "ls1jht",
        "lco1jt", "aseg+DKT_BrainSegVol_pre",
    ]
    scores = df[metadata_cols].copy()
    loadings_rows = []
    variance_rows = []
    manifest_rows = []

    for analysis, pre_columns in definitions.items():
        if not pre_columns:
            raise ValueError(f"No features selected for {analysis}")
        bases = [c[:-4] for c in pre_columns]
        delta_columns = [f"{b}_delta" for b in bases]
        X_pre = df[pre_columns].apply(pd.to_numeric, errors="coerce")
        X_delta = df[delta_columns].apply(pd.to_numeric, errors="coerce")
        X_delta.columns = pre_columns
        if X_pre.isna().any().any() or X_delta.isna().any().any():
            raise ValueError(f"Missing radiomics values in {analysis}")

        variances = X_pre.var(axis=0, ddof=0)
        kept = variances[variances > VARIANCE_THRESHOLD].index.tolist()
        removed = variances[variances <= VARIANCE_THRESHOLD].index.tolist()
        if not kept:
            raise ValueError(f"No features survived variance filtering for {analysis}")

        X_pre = X_pre[kept]
        X_post = X_pre + X_delta[kept]
        scaler = StandardScaler()
        Z_pre = scaler.fit_transform(X_pre)
        Z_post = scaler.transform(X_post)
        pca = PCA(n_components=min(len(df) - 1, len(kept)), svd_solver="full")
        PC_pre = pca.fit_transform(Z_pre)
        PC_post = pca.transform(Z_post)

        signs = np.ones(pca.n_components_)
        signs[pca.components_.sum(axis=1) < 0] = -1
        PC_pre *= signs
        PC_post *= signs
        components = pca.components_ * signs[:, None]

        n_export = min(N_PCS_EXPORT, pca.n_components_)
        for idx in range(n_export):
            pc = f"PC{idx + 1}"
            scores[f"{analysis}_{pc}_pre"] = PC_pre[:, idx]
            scores[f"{analysis}_{pc}_followup"] = PC_post[:, idx]
            scores[f"{analysis}_{pc}_delta"] = PC_post[:, idx] - PC_pre[:, idx]
            variance_rows.append(
                {
                    "analysis": analysis,
                    "component": pc,
                    "n_features": len(kept),
                    "explained_variance_ratio": pca.explained_variance_ratio_[idx],
                    "cumulative_explained_variance_ratio": pca.explained_variance_ratio_[: idx + 1].sum(),
                }
            )

        for feature_idx, feature in enumerate(kept):
            clean = feature[:-4]
            feature_class = "shape" if "_shape_" in feature else "firstorder"
            row = {
                "analysis": analysis,
                "feature": clean,
                "feature_class": feature_class,
                "baseline_mean": scaler.mean_[feature_idx],
                "baseline_sd": scaler.scale_[feature_idx],
            }
            for idx in range(n_export):
                row[f"PC{idx + 1}_loading"] = components[idx, feature_idx]
            loadings_rows.append(row)
            manifest_rows.append(
                {
                    "analysis": analysis,
                    "feature": clean,
                    "feature_class": feature_class,
                    "included": True,
                    "baseline_variance": variances[feature],
                }
            )
        for feature in removed:
            manifest_rows.append(
                {
                    "analysis": analysis,
                    "feature": feature[:-4],
                    "feature_class": "shape" if "_shape_" in feature else "firstorder",
                    "included": False,
                    "baseline_variance": variances[feature],
                }
            )

    scores.to_csv(PCA_PREDEFINED_OUT / "predefined_roi_pca_scores_wide.csv", index=False)
    pd.DataFrame(loadings_rows).to_csv(
        PCA_PREDEFINED_OUT / "predefined_roi_pca_loadings.csv", index=False
    )
    pd.DataFrame(variance_rows).to_csv(
        PCA_PREDEFINED_OUT / "predefined_roi_pca_explained_variance.csv", index=False
    )
    pd.DataFrame(manifest_rows).to_csv(
        PCA_PREDEFINED_OUT / "predefined_roi_pca_feature_manifest.csv", index=False
    )
    print("Predefined ROI baseline-fit PCA completed.")


if __name__ == "__main__":
    main()
