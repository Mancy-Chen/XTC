# Validation of the current manuscript pipeline

Validated on 3 September 2026 using Python 3.12 and the versions recorded in `requirements-tested.txt`.

- All six approved input CSVs passed linkage, unit, dose/change arithmetic, and paired-feature validation for 95 participants.
- All 16 stages in `Code/run_all.py` completed, including regressions, mixed models, 10,000 group-correlation bootstrap resamples, 1,000 PCA bootstrap refits, plots, and supplementary-table exports.
- All 64 comparable CSV outputs matched the verified current-methods rerun (relative tolerance 1e-8; absolute tolerance 1e-10). Execution logs and file/path indexes were excluded from numerical comparison.
- Five input unit tests passed.
- Figure 1 and the whole-brain PC1 loading plot were visually checked after regeneration.
- Anatomical rendering was skipped because the optional spatial atlas was not supplied. Its mapping/status export completed. MRI feature extraction was not rerun.

## Selected numerical checks

| Quantity | Reproduced result |
| --- | --- |
| Partial eta squared, immediate recall | .017491 |
| Partial eta squared, delayed recall | .058018 |
| Selected secondary imaging-memory outcome | Delayed-recall change |
| Adjusted right-hippocampal correlation, XTC-naive | rho = .374371 |
| Adjusted right-hippocampal correlation, XTC users | rho = .107619 |
| Difference, naive minus users | .266751 |
| 95% exploratory bootstrap interval | −.175339 to .604338 |
| Bootstrap outcomes selected after within-group FDR | Right hippocampus only |
| Retained predefined-ROI PCA features | 56 shape; 127 shape plus first-order |
| Retained whole-brain PCA features | 100 regional volumes |

The new behavioral selection JSON explicitly records the partial-eta-squared rule. The supplementary exporter additionally writes the S1 feature-count note and clarifies baseline BrainSegVol, PCA FDR families, and bootstrap interpretation. Those wording changes do not change table values.

Participant-level inputs, intermediate data, and generated output folders remain local and are not committed to the public repository. This validation documents this cohort and environment, not general validation for arbitrary datasets.
