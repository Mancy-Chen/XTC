"""Build a searchable index of all generated output files and manuscript mapping."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import OUTPUT_DIR

MAPPING = [
    ("Full Supplementary Table S2 demographics", "01_full_demographic_table.py", "demographics"),
    ("Behavioral replication", "01_behavioral_replication.py", "OLS_model/behavioral_replication"),
    ("Predefined ROI/BrainSegVol longitudinal LMM", "02_predefined_roi_voxelvolume_lmm.py", "LMM/predefined_roi_voxelvolume"),
    ("ROI raw/adjusted Spearman, normality, selected group-correlation bootstrap", "03_predefined_roi_voxelvolume_behavior.py", "correlations/predefined_roi_voxelvolume"),
    ("Predefined ROI baseline-fit PCA and PC1-PC5 loadings", "04_predefined_roi_pca.py", "PCA/predefined_roi"),
    ("Predefined ROI PC1 LMM, primary and polysubstance", "05_predefined_roi_pca_lmm.py", "LMM/predefined_roi_pca"),
    ("Predefined ROI raw/adjusted Spearman and OLS with/without polysubstance", "06_predefined_roi_pca_behavior.py", "correlations/predefined_roi_pca; OLS_model/predefined_roi_pca"),
    ("Whole-brain baseline-fit VoxelVolume PCA and PC1-PC5 loadings", "07_whole_brain_voxelvolume_pca.py", "PCA/whole_brain"),
    ("Whole-brain PC1 LMM, primary and polysubstance", "08_whole_brain_pca_lmm.py", "LMM/whole_brain_pca"),
    ("Whole-brain raw/adjusted Spearman and OLS", "09_whole_brain_pca_behavior.py", "correlations/whole_brain_pca; OLS_model/whole_brain_pca"),
    ("Whole-brain PCA bootstrap stability", "10_whole_brain_pca_bootstrap.py", "bootstrap/whole_brain_pca"),
    ("All manuscript and supplementary plots", "11_make_plots.py", "plots"),
    ("Supplementary Tables S2-S19 and S1 feature-count note", "14_export_manuscript_tables.py", "manuscript_tables"),
    ("Whole-brain PC1-PC5 anatomical loading projection", "12_spatial_loading_projection.py", "PCA/whole_brain/spatial_projection; plots/whole_brain_pca/spatial_projection"),
]


def main() -> None:
    rows = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUTPUT_DIR)), "folder": str(path.parent.relative_to(OUTPUT_DIR)), "filename": path.name, "extension": path.suffix.lower(), "size_bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "results_file_index.csv", index=False)
    pd.DataFrame(MAPPING, columns=["analysis", "script", "output_location"]).to_csv(OUTPUT_DIR / "manuscript_analysis_mapping.csv", index=False)
    print(f"Results index completed: {len(rows)} generated files indexed.")


if __name__ == "__main__":
    main()
