#!/bin/bash

ID="I162"
session="3"

RAW_DIR="/data/projects/CSC/data/2009-next-mmdewin/2019-next-mmdewin/data_maartje/MR_ruwedata_nifti/3DT1_from_GE_files_fixed_reorientation"
SEG_DIR="/data/projects/CSC/code/XTC/04_SynthSR/test/segmentation"
SYNTH_OUT_DIR="${SEG_DIR}/synthsr_output"
RESULTS_DIR="${SEG_DIR}/segment_output"
SIF="${SEG_DIR}/fastsurfer-gpu.sif"
LICENSE="${SEG_DIR}/license.txt"

INPUT_NII="${RAW_DIR}/${ID}/sessie${session}_3DT1.nii.gz"
SYNTH_NII="${SYNTH_OUT_DIR}/${ID}_sessie${session}_synthsr.nii.gz"
SUBJECT_ID="${ID}_sessie${session}"

module unload freesurfer/7.1.0 || true
module load freesurfer/8.1.0

export FREESURFER_HOME=/opt/amc/freesurfer-8.1.0
set +u
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
set -u

mkdir -p "${SYNTH_OUT_DIR}" "${RESULTS_DIR}"

if [ ! -f "${INPUT_NII}" ]; then
  echo "ERROR: input file not found: ${INPUT_NII}"
  exit 1
fi

if [ ! -f "${SIF}" ]; then
  echo "ERROR: Singularity image not found: ${SIF}"
  exit 1
fi

if [ ! -f "${LICENSE}" ]; then
  echo "ERROR: license not found: ${LICENSE}"
  exit 1
fi

CUDA_VISIBLE_DEVICES="" mri_synthsr \
  --i "${INPUT_NII}" \
  --o "${SYNTH_NII}" \
  --threads 4

singularity exec --nv \
  --no-mount home \
  -B "${SEG_DIR}:/data" \
  "${SIF}" \
  /fastsurfer/run_fastsurfer.sh \
    --fs_license /data/license.txt \
    --t1 "/data/synthsr_output/${ID}_sessie${session}_synthsr.nii.gz" \
    --sid "${SUBJECT_ID}" \
    --sd /data/segment_output \
    --threads 4
