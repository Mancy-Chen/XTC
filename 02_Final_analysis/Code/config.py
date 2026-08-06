"""Central project paths and analysis settings.

All paths are relative to the project folder, so the project can be moved
without editing hard-coded server paths.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "Input"
OUTPUT_DIR = PROJECT_ROOT / "Output"

PREDEFINED_ROI_INPUT = INPUT_DIR / "radiomics_predefine_roi" / "predefined_roi_radiomics_n95_log1p.csv"
WHOLE_BRAIN_INPUT = INPUT_DIR / "radiomics_whole_brain" / "whole_brain_voxelvolume_n95_log1p.csv"
IMAGING_COVARIATES_INPUT = INPUT_DIR / "demographics" / "imaging_covariates_n95_log1p.csv"
BEHAVIOR_INPUT = INPUT_DIR / "demographics" / "behavioral_replication_n95_log1p.csv"
BRAINSEGVOL_INPUT = INPUT_DIR / "demographics" / "brainsegvol_pre_delta_n95.csv"
MERGED_ALL_SAV = INPUT_DIR / "source_original" / "merged_all.sav"

# External FastSurfer segmentation used to project regional PCA loadings.
# Override on another system with the XTC_SPATIAL_ATLAS environment variable.
import os
SPATIAL_ATLAS_PATH = Path(os.environ.get(
    "XTC_SPATIAL_ATLAS",
    "/data/projects/CSC/code/XTC/04_SynthSR/Whole_data/segment_output/"
    "I010_sessie1/mri/aparc.DKTatlas+aseg.deep.withCC.mgz",
))

PCA_PREDEFINED_OUT = OUTPUT_DIR / "PCA" / "predefined_roi"
PCA_WHOLE_BRAIN_OUT = OUTPUT_DIR / "PCA" / "whole_brain"
LMM_ROI_VOLUME_OUT = OUTPUT_DIR / "LMM" / "predefined_roi_voxelvolume"
LMM_PREDEFINED_PCA_OUT = OUTPUT_DIR / "LMM" / "predefined_roi_pca"
LMM_WHOLE_BRAIN_PCA_OUT = OUTPUT_DIR / "LMM" / "whole_brain_pca"
DEMOGRAPHICS_OUT = OUTPUT_DIR / "demographics"
OLS_BEHAVIOR_OUT = OUTPUT_DIR / "OLS_model" / "behavioral_replication"
OLS_PREDEFINED_PCA_OUT = OUTPUT_DIR / "OLS_model" / "predefined_roi_pca"
OLS_WHOLE_BRAIN_PCA_OUT = OUTPUT_DIR / "OLS_model" / "whole_brain_pca"
CORR_ROI_VOLUME_OUT = OUTPUT_DIR / "correlations" / "predefined_roi_voxelvolume"
CORR_PREDEFINED_PCA_OUT = OUTPUT_DIR / "correlations" / "predefined_roi_pca"
CORR_WHOLE_BRAIN_PCA_OUT = OUTPUT_DIR / "correlations" / "whole_brain_pca"
BOOTSTRAP_WHOLE_BRAIN_OUT = OUTPUT_DIR / "bootstrap" / "whole_brain_pca"
PLOT_ROI_VOLUME_OUT = OUTPUT_DIR / "plots" / "predefined_roi_voxelvolume"
PLOT_PREDEFINED_PCA_OUT = OUTPUT_DIR / "plots" / "predefined_roi_pca"
PLOT_WHOLE_BRAIN_PCA_OUT = OUTPUT_DIR / "plots" / "whole_brain_pca"
SPATIAL_PROJECTION_OUT = PCA_WHOLE_BRAIN_OUT / "spatial_projection"
PLOT_SPATIAL_PROJECTION_OUT = PLOT_WHOLE_BRAIN_PCA_OUT / "spatial_projection"
PLOT_BOOTSTRAP_OUT = OUTPUT_DIR / "plots" / "bootstrap"
LOG_DIR = OUTPUT_DIR / "logs"

SUBJECT_COL = "subject_id"
DOSE_COL = "xlttot_sessie3"
LOG_DOSE_COL = "log1p_xtc"
BRAINSEGVOL_COL = "aseg+DKT_BrainSegVol_pre"

POLYSUBSTANCE_COLS = {
    "cannabis": "lca1jt",
    "tobacco": "lsigpw",
    "alcohol": "lalupw",
    "amphetamine": "ls1jht",
    "cocaine": "lco1jt",
}

PREDEFINED_ROIS = [
    "Left_Hippocampus",
    "Right_Hippocampus",
    "Left_Thalamus",
    "Right_Thalamus",
]

ROI_VOLUME_COLUMNS = {
    "Left hippocampus": "left_hippocampus",
    "Right hippocampus": "right_hippocampus",
    "Left thalamus": "left_thalamus",
    "Right thalamus": "right_thalamus",
}

N_PCS_EXPORT = 5
VARIANCE_THRESHOLD = 1e-12
RANDOM_SEED = 20260806
N_PERMUTATIONS = 100000
N_BOOTSTRAP = 1000

GROUP_ORDER = ["XTC-naive", "XTC users"]
COLORS = {"XTC users": "#eb5600", "XTC-naive": "#1a9988"}
