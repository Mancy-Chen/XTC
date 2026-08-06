# XTC: Longitudinal MRI Radiomics and Verbal-Memory Analysis

This repository contains the reproducible analysis pipeline for a longitudinal substudy of incident XTC/MDMA exposure in 95 young adults from the NeXT cohort.

The project integrates behavioral, demographic, substance-use, predefined-region radiomics, and whole-brain volumetric MRI data to investigate whether incident XTC exposure is associated with changes in verbal memory and whether quantitative structural brain features—particularly from the hippocampus and thalamus—capture those changes.

## Research aims

The pipeline is designed to:

- replicate previously reported associations between incident XTC exposure and RAVLT verbal-memory change;
- quantify longitudinal VoxelVolume changes in the bilateral hippocampus and thalamus;
- summarize correlated predefined-ROI radiomics features using principal component analysis (PCA);
- summarize distributed whole-brain volumetric variation using baseline-fit PCA;
- test longitudinal time-by-XTC-dose effects with linear mixed-effects models;
- examine raw and BrainSegVol-adjusted associations between imaging change and delayed-recall change;
- assess whether imaging measures add explanatory value beyond baseline memory, XTC dose, demographic variables, and polysubstance use;
- evaluate PCA robustness using permutation tests and bootstrap loading stability;
- generate manuscript and supplementary tables, figures, loading plots, and spatial projections.

## Analysis overview

The main workflow includes:

1. input validation and participant-level consistency checks;
2. demographic and substance-use summaries;
3. behavioral replication analyses;
4. predefined hippocampal and thalamic VoxelVolume analyses;
5. predefined-ROI radiomics PCA, mixed-effects models, and behavioral associations;
6. whole-brain VoxelVolume PCA, mixed-effects models, and behavioral associations;
7. permutation testing and PCA bootstrap stability analysis;
8. manuscript-quality plots, PC1–PC5 loading figures, and spatial loading projections;
9. automated indexing of generated results.

## Repository structure

```text
XTC/
├── Code/
│   ├── 00_validate_inputs.py
│   ├── 01_full_demographic_table.py
│   ├── 01_behavioral_replication.py
│   ├── 02_predefined_roi_voxelvolume_lmm.py
│   ├── 03_predefined_roi_voxelvolume_behavior.py
│   ├── 04_predefined_roi_pca.py
│   ├── 05_predefined_roi_pca_lmm.py
│   ├── 06_predefined_roi_pca_behavior.py
│   ├── 07_whole_brain_voxelvolume_pca.py
│   ├── 08_whole_brain_pca_lmm.py
│   ├── 09_whole_brain_pca_behavior.py
│   ├── 10_whole_brain_pca_bootstrap.py
│   ├── 11_make_plots.py
│   ├── 12_spatial_loading_projection.py
│   ├── 13_build_results_index.py
│   ├── config.py
│   ├── utils.py
│   └── run_all.py
├── Input/        # Expected analysis-ready data; not distributed publicly
├── Output/       # Generated tables, models, correlations, plots, and logs
├── README.md
└── LICENSE
```

All project paths are defined relative to the repository root in `Code/config.py`, allowing the project folder to be moved without editing hard-coded analysis paths. The optional spatial projection step can use the `XTC_SPATIAL_ATLAS` environment variable to specify a local FastSurfer/FreeSurfer segmentation.

## Environment

A dedicated conda environment is recommended:

```bash
conda create -n xtc-final -c conda-forge \
    python=3.10 \
    "numpy>=1.26,<2" \
    pandas scipy scikit-learn statsmodels matplotlib \
    pyreadstat nibabel \
    -y

conda activate xtc-final
```

NumPy is constrained to version 1.x for compatibility with scientific Python packages that may have been compiled against the NumPy 1.x binary interface.

## Running the pipeline

From the repository root:

```bash
cd Code
python run_all.py
```

The runner executes all analysis scripts in sequence and stops if any step fails. A complete log is written to:

```text
Output/logs/run_all.log
```

Individual modules can also be run separately, for example:

```bash
python 11_make_plots.py
```

## Expected inputs

The analysis expects restricted participant-level datasets in the following folders:

```text
Input/demographics/
Input/radiomics_predefine_roi/
Input/radiomics_whole_brain/
Input/source_original/
```

The participant-level data are not included in this repository because they contain restricted research information. Users must provide compatible analysis-ready files or modify the corresponding paths in `Code/config.py`.

## Main outputs

The pipeline writes results to organized subfolders under `Output/`, including:

- demographic tables;
- behavioral regression results;
- predefined-ROI and whole-brain PCA scores, loadings, and explained variance;
- linear mixed-effects model summaries;
- raw and adjusted Spearman correlations;
- fully adjusted OLS models and nested-model comparisons;
- permutation-test results;
- bootstrap PCA stability summaries;
- publication-quality figures and PC1–PC5 loading plots;
- a machine-readable index of generated result files.

## Reproducibility notes

- PCA is fitted using baseline data and follow-up observations are projected into the baseline feature space.
- XTC dose and polysubstance covariates use `ln(1 + x)` transformations where specified.
- Random procedures use a fixed seed defined in `Code/config.py`.
- The default pipeline uses 100,000 permutations and 1,000 bootstrap resamples.
- Group colors are fixed as orange for XTC users (`#eb5600`) and green for XTC-naive participants (`#1a9988`).

## Scope

This repository supports the statistical analyses used for the accompanying manuscript and supplementary material. It is research code rather than a clinical diagnostic tool, and its outputs should not be interpreted as individual-level predictions of cognitive impairment.

## License

See [LICENSE](LICENSE).
