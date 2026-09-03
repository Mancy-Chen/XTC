# Current manuscript analysis

This pipeline reproduces the final NeXT imaging analysis in 95 participants (48 XTC-naive; 47 incident users). It uses the six approved CSV inputs and writes all results locally under `Output/`.

## Inputs and execution

```text
Input/
  demographics/
    behavioral_replication_n95_log1p.csv
    imaging_covariates_n95_log1p.csv
    brainsegvol_pre_delta_n95.csv
    demographics_n95_deidentified.csv
  radiomics_predefine_roi/
    predefined_roi_radiomics_n95_log1p.csv
  radiomics_whole_brain/
    whole_brain_voxelvolume_n95_log1p.csv
```

All files use reassigned IDs XTC001–XTC095. The behavioral key is `studnr`; other files use `subject_id`. Do not substitute a convenience merged CSV for these inputs. The input validator checks linkage, dose and change-score arithmetic, units, and feature pairing. Original IDs and SPSS files are not needed; `sav_legacy_reader.py` is retained only for historical reference.

Python 3.12 was used for validation. From this folder:

```bash
python -m pip install -r requirements.txt
python Code/run_all.py
python -m unittest discover -s Code -p 'test_*.py'
```

`requirements-tested.txt` records the validation environment. `Output/logs/run_all.log` records each stage and stops on failure. Use a fresh `Output/` directory when upgrading from older releases: old permutation and nested-model files are not outputs of this version and should not be interpreted as current results.

## Behavioral replication and outcome selection

Immediate- and delayed-recall change are follow-up minus baseline. ANCOVAs adjust for baseline performance, age, sex, IQ, and cannabis, tobacco, alcohol, amphetamine, and cocaine use. Recognition decline is follow-up below baseline and is analyzed with logistic regression using the corresponding covariates.

The larger adjusted XTC-group partial eta squared among immediate and delayed recall determines the outcome for subsequent **secondary** imaging–memory analyses. Recognition odds ratios are not compared with partial eta squared. The choice and both candidate effect sizes are exported to `OLS_model/behavioral_replication/memory_outcome_selection.json`. In this cohort, delayed recall is selected. The imaging inputs contain delayed-recall fields; the runner stops if new behavioral data select immediate recall, rather than silently analyzing the wrong outcome.

## Regional and global volumes

The four ROIs are left/right hippocampus and thalamus. Participant-random-intercept models estimate within-group session effects and between-group session-by-XTC-group interactions. Regional models include sex and **session-specific** BrainSegVol. The separate BrainSegVol outcome adjusts for sex. BH FDR is applied across the four ROIs and BrainSegVol separately for each within-group family and the interaction family.

Raw and residualized volume changes are correlated with memory change using Spearman correlations in the full sample and each XTC group. ROI change is residualized for baseline ROI volume and baseline BrainSegVol; BrainSegVol change for baseline BrainSegVol alone. Residualization uses the pooled sample. BH FDR is applied across the five imaging outcomes within each sample and adjustment type. Shapiro–Wilk results for raw/adjusted imaging changes and memory change are exported for description; Spearman correlations are used consistently.

Only imaging outcomes whose adjusted association has q < .05 in either XTC group enter the exploratory between-group comparison. Here that selects only right hippocampus. The code draws 10,000 participant-level resamples with replacement separately within groups, repeats pooled residualization in every resample, and computes rho_naive minus rho_users. The reported 95% percentile intervals are unadjusted and exploratory; they do not account for outcome selection. No additional FDR adjustment is applied to these intervals. Group-comparison seed: 20260901; numerical sorting makes resampling independent of pseudonym IDs and row order. The current interval is −.175 to .604, with difference .267.

## PCA and imaging–memory models

PCA is fitted separately to 56 combined four-ROI shape features, 127 combined shape/first-order features (56 shape + 71 first-order), and 100 whole-brain regional volumes. The left-hippocampal first-order 10th-percentile change feature is excluded because its baseline counterpart is unavailable. Four hypothalamic regions were excluded upstream; baseline zero/near-zero variance filtering is also applied. Feature manifests identify the retained variables.

Each scaler and PCA is fitted exclusively to baseline data; follow-up uses the baseline means, standard deviations, and loadings. Signs are aligned consistently. Inferential analyses focus on PC1; PC2–PC5 are exported for descriptive variance/loading figures only.

PC1 mixed models use:

```text
PC1 ~ time * ln(1 + XTC dose) + age + sex + baseline BrainSegVol + (1 | participant)
```

Sensitivity models add cannabis, tobacco, alcohol, amphetamine, and cocaine use. PC1–memory correlations are computed in the full sample and each group, both raw and after residualizing PC1 change for baseline PC1 and baseline BrainSegVol. BH FDR is across the **two predefined-ROI PCA feature sets**, separately within each sample and adjustment type. Whole-brain PC1 is a separate single-outcome analysis.

Participant-level OLS models use:

```text
Follow-up delayed recall ~ PC1 change + baseline PC1 + baseline delayed recall
    + baseline BrainSegVol + ln(1 + XTC dose) + age + sex + IQ + other substance use
```

Predefined-ROI models with and without other substance covariates are exported; the fully adjusted models are emphasized. There is no nested cross-validation, nested/reduced-model comparison, incremental-R² analysis, or partial F-test comparing PCA models.

Whole-brain PCA stability uses 1,000 baseline participant resamples (seed 20260806), refitting the scaler/PCA, aligning PC1 signs, and projecting the original baseline and follow-up observations. Its percentile ranges reflect PCA-refitting variability rather than confidence intervals for population correlations or external generalizability.

## Covariates and missing data

The input substance-use variables are already ln(1 + x)-transformed; do not transform them again. Continuous PC1-model predictors are mean-centered, and sex is categorical. Four imaging-age values are mean-imputed in models requiring this covariate. Behavioral and imaging age are distinct source fields; behavioral ages are not substituted for imaging ages. The complete-covariate description therefore has an imaging-age exception.

The dedicated BrainSegVol CSV stores mm³; baseline BrainSegVol in imaging inputs is in cm³. The code converts units consistently. Baseline BrainSegVol is used in PCA models; concurrent BrainSegVol is used in longitudinal ROI models. Neither is an ICV adjustment.

## Figures, tables, and optional atlas

See [MANUSCRIPT_OUTPUTS.md](MANUSCRIPT_OUTPUTS.md) for exact output paths and captions. Figure 1 uses “Change in RAVLT delayed recall (words),” lists XTC-naive above XTC users, and has no embedded caption. Its lines and bands remain descriptive OLS fits, not group-slope tests. Figure S2 labels use LH/RH/LT/RT, defined in the separate caption. Figure S3A uses full anatomical names.

An atlas is not distributed. To generate spatial maps (Figure S3B), provide `Input/spatial_atlas/atlas.mgz` or set `XTC_SPATIAL_ATLAS` to an appropriate FastSurfer segmentation and install `nibabel`. Without it, that stage exports its mapping/status and skips anatomical rendering; numerical analyses and the other figures still run.
