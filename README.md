# XTC: Longitudinal MRI Radiomics and Verbal-Memory Analysis

This repository contains feature-extraction and statistical-analysis code for a longitudinal substudy of incident XTC/MDMA exposure in 95 young adults from the NeXT cohort.

The project combines verbal-memory outcomes, demographic and substance-use covariates, predefined hippocampal/thalamic radiomics, and whole-brain volumetric MRI measures to investigate whether incident XTC exposure is associated with longitudinal memory change and whether structural MRI features capture corresponding brain changes.

## Repository overview

The repository is organized into two main stages:

1. **`01_Radiomics_extract_Features/`** — scripts used to derive radiomics and volumetric features from FastSurfer/FreeSurfer outputs.
2. **`02_Final_analysis/Code/`** — the manuscript-oriented statistical pipeline operating on analysis-ready participant-level datasets.

Restricted participant-level data and generated outputs are not distributed in this public repository.

## Research aims

The analysis code is designed to:

- replicate previously reported associations between incident XTC exposure and RAVLT verbal-memory change;
- quantify longitudinal VoxelVolume change in the bilateral hippocampus and thalamus;
- examine corresponding global brain-volume change using BrainSegVol;
- summarize correlated predefined-ROI radiomics features using principal component analysis (PCA);
- summarize distributed whole-brain VoxelVolume variation using baseline-fit PCA;
- test longitudinal time-by-XTC-dose effects with linear mixed-effects models;
- examine raw and adjusted associations between imaging change and delayed-recall change;
- compare subgroup imaging–behavior associations using permutation testing where implemented;
- assess whether PCA-derived imaging measures add explanatory value beyond baseline memory, XTC exposure, demographic variables, global brain volume, and polysubstance use;
- evaluate the stability of the whole-brain PC1 solution using bootstrap resampling;
- generate manuscript and supplementary tables, figures, loading plots, spatial loading projections, and an index of result files.

## Repository structure

```text
XTC/
├── 01_Radiomics_extract_Features/
│   ├── Extract_first_order_shape_features.py
│   ├── Extract_voxel_volume_whole_brain.py
│   └── Extract_whole_brain_volume.py
│
├── 02_Final_analysis/
│   ├── Code/
│   │   ├── 00_validate_inputs.py
│   │   ├── 01_full_demographic_table.py
│   │   ├── 01_behavioral_replication.py
│   │   ├── 02_predefined_roi_voxelvolume_lmm.py
│   │   ├── 03_predefined_roi_voxelvolume_behavior.py
│   │   ├── 04_predefined_roi_pca.py
│   │   ├── 05_predefined_roi_pca_lmm.py
│   │   ├── 06_predefined_roi_pca_behavior.py
│   │   ├── 07_whole_brain_voxelvolume_pca.py
│   │   ├── 08_whole_brain_pca_lmm.py
│   │   ├── 09_whole_brain_pca_behavior.py
│   │   ├── 10_whole_brain_pca_bootstrap.py
│   │   ├── 11_make_plots.py
│   │   ├── 12_spatial_loading_projection.py
│   │   ├── 13_build_results_index.py
│   │   ├── config.py
│   │   ├── sav_legacy_reader.py
│   │   ├── utils.py
│   │   └── run_all.py
│   │
│   ├── Input/      # local/restricted; not distributed on GitHub
│   └── Output/     # generated locally; not distributed on GitHub
│
├── .gitignore
├── README.md
└── LICENSE
```

## Stage 1: MRI feature extraction

The scripts in `01_Radiomics_extract_Features/` are preprocessing/feature-extraction utilities and are **not** executed by `02_Final_analysis/Code/run_all.py`.

### `Extract_first_order_shape_features.py`

Extracts PyRadiomics **shape** and **first-order** features from four predefined FastSurfer regions:

- left thalamus;
- left hippocampus;
- right thalamus;
- right hippocampus.

The script converts MGZ images to SimpleITK images, aligns segmentation masks to the image geometry where necessary, creates binary ROI masks, and exports participant/session-level radiomics features.

### `Extract_voxel_volume_whole_brain.py`

Extracts PyRadiomics **VoxelVolume** across non-background regions in the FastSurfer `aparc.DKTatlas+aseg.deep.withCC.mgz` segmentation. It uses the FreeSurfer lookup table when available and produces long- and wide-format volumetric datasets together with a record of failed extractions.

### `Extract_whole_brain_volume.py`

Parses FastSurfer/FreeSurfer `.stats` files and exports regional volume tables and global measures such as BrainSegVol in long and wide formats.

### Paths for extraction scripts

The extraction scripts currently contain project-specific path settings near the top of each file. Before running them on another system, update the image, segmentation-mask, and output paths to match the local environment.

Typical dependencies for this stage include:

```bash
pip install numpy pandas nibabel SimpleITK pyradiomics
```

FastSurfer/FreeSurfer outputs are expected to have been generated separately.

## Stage 2: final statistical analysis

The final analysis pipeline is located in:

```text
02_Final_analysis/Code/
```

`config.py` defines the project-relative input and output locations. Because `PROJECT_ROOT` is derived from the location of `config.py`, the `02_Final_analysis` folder can be moved without editing the statistical-analysis paths.

### Analysis sequence

| Script | Purpose |
|---|---|
| `00_validate_inputs.py` | Validates required datasets, columns, participant IDs, and basic consistency before analysis. |
| `01_full_demographic_table.py` | Produces demographic and substance-use summaries for the final imaging sample. |
| `01_behavioral_replication.py` | Replicates NeXT RAVLT findings for immediate recall, delayed recall, and recognition decline using adjusted regression models. |
| `02_predefined_roi_voxelvolume_lmm.py` | Fits longitudinal mixed-effects models for hippocampal/thalamic VoxelVolume and BrainSegVol, including within-group and direct group-change comparisons. |
| `03_predefined_roi_voxelvolume_behavior.py` | Tests raw and adjusted associations between ROI/BrainSegVol change and delayed-recall change, including subgroup analyses, FDR correction, and permutation tests of subgroup-correlation differences. |
| `04_predefined_roi_pca.py` | Fits baseline-derived PCA models to predefined-ROI shape-only and shape + first-order radiomics features and projects follow-up data into the baseline PCA space. |
| `05_predefined_roi_pca_lmm.py` | Tests time-by-XTC-dose effects on predefined-ROI PC1 scores using primary and selected polysubstance-adjusted mixed-effects models. |
| `06_predefined_roi_pca_behavior.py` | Examines raw/adjusted PC1–memory associations and OLS models for predefined-ROI PCA measures. |
| `07_whole_brain_voxelvolume_pca.py` | Fits baseline-derived PCA to paired whole-brain VoxelVolume features and exports PC scores, loadings, explained variance, and the retained-feature manifest. |
| `08_whole_brain_pca_lmm.py` | Fits primary and polysubstance-adjusted mixed-effects models for whole-brain PC1, including the time-by-log-XTC-dose interaction. |
| `09_whole_brain_pca_behavior.py` | Tests raw/adjusted PC1–memory associations, fully adjusted OLS models, and the incremental value of PC1 beyond covariates. |
| `10_whole_brain_pca_bootstrap.py` | Evaluates PC1 stability by refitting baseline PCA in bootstrap samples, sign-aligning loadings, and projecting the original baseline/follow-up data. |
| `11_make_plots.py` | Generates manuscript- and supplementary-quality figures from the analysis outputs. |
| `12_spatial_loading_projection.py` | Projects PC1–PC5 regional loadings into a FastSurfer segmentation atlas and generates spatial loading visualizations. |
| `13_build_results_index.py` | Builds an index of generated output files. |
| `sav_legacy_reader.py` | Legacy SPSS reader retained for reference; not used by the deidentified pipeline. |
| `utils.py` | Shared data-processing, modeling, FDR, export, and utility functions. |
| `run_all.py` | Runs the complete final-analysis sequence in order and stops if a step fails. |

## Expected analysis inputs

`02_Final_analysis/Code/config.py` expects the following local files:

```text
02_Final_analysis/Input/
├── demographics/
│   ├── imaging_covariates_n95_log1p.csv
│   ├── behavioral_replication_n95_log1p.csv
│   ├── brainsegvol_pre_delta_n95.csv
│   └── demographics_n95_deidentified.csv
├── radiomics_predefine_roi/
│   └── predefined_roi_radiomics_n95_log1p.csv
├── radiomics_whole_brain/
    └── whole_brain_voxelvolume_n95_log1p.csv
```

These six participant-level CSVs are restricted research data and are intentionally not included in the public repository. Obtain the deidentified data package through the study's approved data-sharing process and copy its `Input/` folder into `02_Final_analysis/`. The earlier data-only package requires its five `analysis_inputs/` CSVs plus `supplementary/demographics_n95_deidentified.csv` placed under `Input/demographics/`.

All files must use the same reassigned IDs (`XTC001`–`XTC095`). The behavioral file uses the key `studnr`; other files use `subject_id`. The validator checks linkage, outcome changes, dose transformations and BrainSegVol units without referencing original IDs. The demographic table reads the demographic CSV; no raw SPSS file is required. The single combined convenience CSV is not a replacement for these six inputs.

Source-format units are preserved: the dedicated BrainSegVol CSV uses mm³; the baseline BrainSegVol covariate in other imaging inputs uses cm³. The code handles this conversion. BrainSegVol is not ICV.

## Environment for the final analysis

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r 02_Final_analysis/requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Install `nibabel` separately only if generating anatomical loading maps. MRI feature extraction has its own dependencies listed above.

## Running the final pipeline

From the repository root:

```bash
cd 02_Final_analysis/Code
python run_all.py
```

The runner executes the final statistical-analysis scripts sequentially. If a step fails, execution stops and the failing step is reported.

A combined log is written to:

```text
02_Final_analysis/Output/logs/run_all.log
```

Individual analyses can also be run separately, for example:

```bash
cd 02_Final_analysis/Code
python 08_whole_brain_pca_lmm.py
```

## PCA implementation

Both predefined-ROI and whole-brain PCA use a **baseline-fit / follow-up-projection** design:

1. baseline features are variance-filtered;
2. the scaler is fitted on baseline data;
3. PCA is fitted on standardized baseline data;
4. follow-up features are transformed with the baseline scaler;
5. follow-up observations are projected onto the baseline-derived PCA axes;
6. longitudinal PC change is calculated as follow-up score minus baseline score.

The pipeline exports the first five PCs where available, together with explained variance and regional/feature loadings.

## Longitudinal and behavioral models

The analysis includes several complementary model families:

- linear mixed-effects models for longitudinal regional volumes and PC scores;
- time-by-XTC-dose interaction models using `ln(1 + XTC dose)`;
- polysubstance-adjusted sensitivity models;
- raw and residualized Spearman correlations with delayed-recall change;
- FDR correction for predefined families of correlation/ROI tests;
- label-permutation tests for differences between XTC-naive and XTC-user subgroup correlations in the predefined VoxelVolume analysis;
- OLS/ANCOVA-style models in which follow-up delayed recall is modeled while retaining baseline delayed recall as a covariate;
- nested-model comparisons for the incremental explanatory value of whole-brain PC1.

Polysubstance covariates use `ln(1 + x)` transformed variables in the analysis-ready input datasets where specified.

## PCA robustness

Whole-brain PC1 robustness is evaluated with bootstrap resampling. For each bootstrap iteration, the PCA axis is refitted using resampled baseline participants, aligned to the original PC1 direction, and used to project the complete original baseline and follow-up datasets.

The bootstrap analysis summarizes quantities including:

- PC1 explained variance;
- loading cosine similarity;
- baseline and longitudinal PC1 score reproducibility;
- overlap among the strongest regional loadings;
- stability of imaging–memory associations;
- loading percentile intervals and sign consistency.

The default configuration uses:

```text
100,000 permutations
1,000 bootstrap resamples
random seed = 20260806
```

## Spatial loading projection

`12_spatial_loading_projection.py` maps whole-brain PC1–PC5 regional loadings back into a FastSurfer `aparc.DKTatlas+aseg.deep.withCC.mgz` segmentation.

The optional atlas defaults to `02_Final_analysis/Input/spatial_atlas/atlas.mgz`. It can be overridden with:

```bash
export XTC_SPATIAL_ATLAS=/path/to/aparc.DKTatlas+aseg.deep.withCC.mgz
```

If the atlas is unavailable, the script records the status and exits without preventing the rest of the portable pipeline from completing.

## Main outputs

Generated results are written beneath:

```text
02_Final_analysis/Output/
```

The pipeline organizes results into folders for:

- demographics;
- behavioral replication models;
- predefined-ROI VoxelVolume mixed models and correlations;
- predefined-ROI PCA scores, loadings, mixed models, correlations, and OLS models;
- whole-brain PCA scores, loadings, explained variance, mixed models, correlations, and OLS models;
- permutation-test summaries;
- whole-brain PCA bootstrap stability;
- manuscript and supplementary plots;
- spatial loading projections;
- logs and a generated results index.

## Reproducibility notes

- The final imaging sample is designed around a consistent set of 95 participants.
- PCA is fitted at baseline; follow-up data are projected into the baseline feature space rather than refitting PCA at follow-up.
- XTC dose is represented as `ln(1 + dose)` in the final models.
- Polysubstance sensitivity covariates use corresponding log1p-transformed variables where specified.
- Age is mean-imputed where implemented by the analysis scripts.
- Random procedures use the fixed seed in `config.py`.
- Group plotting colors are fixed as orange for XTC users (`#eb5600`) and green for XTC-naive participants (`#1a9988`).
- Participant-level input data and generated output directories are excluded from the public repository.

## Data availability and scope

This repository provides research code for reproducibility of the associated XTC/MDMA longitudinal MRI analyses. Participant-level data are not publicly distributed because they contain restricted research information.

The code is intended for research and manuscript reproducibility. It is not a clinical diagnostic tool and should not be used to make individual-level clinical predictions.

## Deidentified-input adaptation and validation

The code now supports the deidentified CSV package. Statistical model formulas, feature-selection rules, covariates, FDR families and resampling settings are unchanged. `utils.py` loads statsmodels only when model/FDR helpers are called.

Verified for this adaptation: all code compiled; five input tests passed; the demographic table matched the archived table exactly; PCA explained variance and loadings matched within numerical tolerance; all 1,000 bootstrap resamples completed. Regression and mixed-effects stages were not rerun because statsmodels installation was blocked in the verification environment. Anatomical maps were skipped because an atlas was not supplied. This is not a claim that the complete adapted pipeline has been rerun.

Run input tests after placing the approved data locally:

```bash
python -m unittest discover -s 02_Final_analysis/Code -p 'test_*.py'
```

Known source-data issues are retained and reported: four missing imaging ages; different behavioral and imaging age values; and a left hippocampal first-order 10th-percentile change column with no baseline counterpart. The existing paired-feature PCA selection excludes that unpaired feature. Do not silently substitute age sources. Reordered deidentified participants may produce small differences in random resampling results even with the same seed.

The demographic CSV contains baseline IQ only, so the auxiliary demographic summary leaves follow-up IQ missing; the main table already displays follow-up IQ as NA. Raw SPSS reading is no longer part of the active pipeline.

## License

See [LICENSE](LICENSE).
