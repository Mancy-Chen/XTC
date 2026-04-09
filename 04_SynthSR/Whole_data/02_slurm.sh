#!/bin/bash
#SBATCH --job-name=synthsr_fs_all
#SBATCH --output=/data/projects/CSC/code/XTC/04_SynthSR/Whole_data/logs/synthsr_fs_%A_%a.out
#SBATCH --error=/data/projects/CSC/code/XTC/04_SynthSR/Whole_data/logs/synthsr_fs_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=1-421

set -e
set -u
set -o pipefail

LIST="/data/projects/CSC/code/XTC/04_SynthSR/Whole_data/input_list.txt"
SEG_DIR="/data/projects/CSC/code/XTC/04_SynthSR/Whole_data"
TOP_DIR="/data/projects/CSC/code/XTC/04_SynthSR"

SYNTH_OUT_DIR="${SEG_DIR}/synthsr_output"
RESULTS_DIR="${SEG_DIR}/segment_output"
SIF="${TOP_DIR}/fastsurfer-gpu.sif"
LICENSE="${TOP_DIR}/license.txt"

mkdir -p "${SEG_DIR}/logs" "${SYNTH_OUT_DIR}" "${RESULTS_DIR}"

INPUT_NII=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")

if [ -z "${INPUT_NII}" ]; then
  echo "ERROR: no input found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

SUBJECT_ID=$(basename "$(dirname "$INPUT_NII")")
FILENAME=$(basename "$INPUT_NII")

TIMEPOINT=$(echo "$FILENAME" | sed -E 's/^(sessie[0-9]+)_3DT1\.nii\.gz$/\1/')
SESSION_NUM=$(echo "$TIMEPOINT" | sed -E 's/^sessie([0-9]+)$/\1/')

if [ -z "$TIMEPOINT" ] || [ -z "$SESSION_NUM" ]; then
  echo "ERROR: could not parse timepoint from filename: $FILENAME"
  exit 1
fi

RUN_ID="${SUBJECT_ID}_${TIMEPOINT}"
SYNTH_NII="${SYNTH_OUT_DIR}/${RUN_ID}_synthsr.nii.gz"

echo "INPUT_NII=${INPUT_NII}"
echo "SUBJECT_ID=${SUBJECT_ID}"
echo "TIMEPOINT=${TIMEPOINT}"
echo "SESSION_NUM=${SESSION_NUM}"
echo "RUN_ID=${RUN_ID}"
echo "SYNTH_NII=${SYNTH_NII}"

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

if [ -d "${RESULTS_DIR}/${RUN_ID}" ]; then
  echo "Skipping ${RUN_ID}: FastSurfer output already exists"
  exit 0
fi

module unload freesurfer/7.1.0 || true
module load freesurfer/8.1.0 || true

unset FSFAST_HOME
unset SUBJECTS_DIR
unset FUNCTIONALS_DIR
unset MNI_DIR
unset FSL_DIR
unset FREESURFER_HOME
unset FMRI_ANALYSIS_DIR

export FREESURFER_HOME=/opt/amc/freesurfer-8.1.0

set +e
set +u
set +o pipefail
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
FS_SOURCE_STATUS=$?
set -e
set -u
set -o pipefail

if [ $FS_SOURCE_STATUS -ne 0 ]; then
  echo "ERROR: failed to source FreeSurfer"
  exit 1
fi

if [ -f "${SYNTH_NII}" ]; then
  echo "SynthSR output already exists, skipping SynthSR: ${SYNTH_NII}"
else
  CUDA_VISIBLE_DEVICES="" mri_synthsr \
    --i "${INPUT_NII}" \
    --o "${SYNTH_NII}" \
    --threads "${SLURM_CPUS_PER_TASK}"
fi

if [ ! -f "${SYNTH_NII}" ]; then
  echo "ERROR: SynthSR output not created: ${SYNTH_NII}"
  exit 1
fi

singularity exec --nv \
  --no-mount home \
  -B "${SEG_DIR}:/data" \
  -B "${LICENSE}:/extra/license.txt" \
  "${SIF}" \
  /fastsurfer/run_fastsurfer.sh \
    --fs_license /extra/license.txt \
    --t1 "/data/synthsr_output/${RUN_ID}_synthsr.nii.gz" \
    --sid "${RUN_ID}" \
    --sd /data/segment_output \
    --threads "${SLURM_CPUS_PER_TASK}"

echo "Finished ${RUN_ID}"
