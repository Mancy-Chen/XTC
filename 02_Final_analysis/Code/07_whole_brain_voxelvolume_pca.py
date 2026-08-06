"""Baseline-fit PCA of whole-brain PyRadiomics VoxelVolume features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import N_PCS_EXPORT, PCA_WHOLE_BRAIN_OUT, VARIANCE_THRESHOLD, WHOLE_BRAIN_INPUT
from utils import ensure_dirs, read_csv_numeric


def identify_features(df: pd.DataFrame) -> list[str]:
    excluded = {"vwrec_pre", "aseg+DKT_BrainSegVol_pre"}
    features = [
        c for c in df.columns
        if c.endswith("_pre")
        and c not in excluded
        and f"{c[:-4]}_delta" in df.columns
    ]
    if not features:
        raise ValueError("No paired whole-brain pre/delta VoxelVolume features found.")
    return features


def main() -> None:
    ensure_dirs([PCA_WHOLE_BRAIN_OUT])
    df = read_csv_numeric(WHOLE_BRAIN_INPUT)
    pre_columns = identify_features(df)
    bases = [c[:-4] for c in pre_columns]
    delta_columns = [f"{b}_delta" for b in bases]

    X_pre = df[pre_columns].apply(pd.to_numeric, errors="coerce")
    X_delta = df[delta_columns].apply(pd.to_numeric, errors="coerce")
    X_delta.columns = pre_columns
    if X_pre.isna().any().any() or X_delta.isna().any().any():
        raise ValueError("Missing values found in whole-brain PCA feature pairs.")

    variances = X_pre.var(axis=0, ddof=0)
    kept = variances[variances > VARIANCE_THRESHOLD].index.tolist()
    removed = variances[variances <= VARIANCE_THRESHOLD].index.tolist()
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

    metadata_cols = [
        "subject_id", "sex", "age", "xlttot_sessie3", "log1p_xtc", "iq",
        "vwrec_pre", "vwrec_delta", "lca1jt", "lsigpw", "lalupw", "ls1jht",
        "lco1jt", "aseg+DKT_BrainSegVol_pre",
    ]
    scores = df[metadata_cols].copy()
    n_export = min(N_PCS_EXPORT, pca.n_components_)
    for idx in range(n_export):
        pc = f"PC{idx + 1}"
        scores[f"{pc}_pre"] = PC_pre[:, idx]
        scores[f"{pc}_followup"] = PC_post[:, idx]
        scores[f"{pc}_delta"] = PC_post[:, idx] - PC_pre[:, idx]
    scores.to_csv(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_scores_wide.csv", index=False)

    loadings = pd.DataFrame({
        "feature": [c[:-4] for c in kept],
        "baseline_mean_cm3": scaler.mean_,
        "baseline_sd_cm3": scaler.scale_,
        "baseline_variance_cm6": variances.loc[kept].to_numpy(),
    })
    for idx in range(n_export):
        loadings[f"PC{idx + 1}_loading"] = components[idx]
    loadings.to_csv(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_loadings.csv", index=False)

    explained = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        "orientation_sign": signs,
    })
    explained.to_csv(
        PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_explained_variance.csv", index=False
    )

    manifest = pd.DataFrame({
        "feature": [c[:-4] for c in pre_columns],
        "included": [c in kept for c in pre_columns],
        "baseline_variance": [variances[c] for c in pre_columns],
    })
    manifest.to_csv(PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_feature_manifest.csv", index=False)
    print(
        f"Whole-brain baseline-fit PCA completed: {len(kept)} features retained; "
        f"{len(removed)} removed."
    )


if __name__ == "__main__":
    main()
