#!/bin/bash
#SBATCH --job-name=bugs_fs
#SBATCH --output=/data/projects/CSC/code/XTC/04_SynthSR/Bugs/logs/bugs_%A_%a.out
#SBATCH --error=/data/projects/CSC/code/XTC/04_SynthSR/Bugs/logs/bugs_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=1-3

set -e
set -u
set -o pipefail
set -x

BASE_DIR="/data/projects/CSC/code/XTC/04_SynthSR/Bugs"
TOP_DIR="/data/projects/CSC/code/XTC/04_SynthSR"

LIST="${BASE_DIR}/bugs_input_list.txt"
OUTPUT_ROOT="${BASE_DIR}/Output"
SYNTH_OUT_DIR="${OUTPUT_ROOT}/synthsr_output"
SEG_OUT_DIR="${OUTPUT_ROOT}/segment_output"
LOG_DIR="${BASE_DIR}/logs"

SIF="${TOP_DIR}/fastsurfer-gpu.sif"
LICENSE="${TOP_DIR}/license.txt"

mkdir -p "${SYNTH_OUT_DIR}" "${SEG_OUT_DIR}" "${LOG_DIR}"

INPUT_NII=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${LIST}")

if [ -z "${INPUT_NII}" ]; then
  echo "ERROR: no input found for task ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

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

RUN_ID=$(basename "$(dirname "$(dirname "${INPUT_NII}")")")
SYNTH_NII="${SYNTH_OUT_DIR}/${RUN_ID}_synthsr.nii.gz"

echo "INPUT_NII=${INPUT_NII}"
echo "RUN_ID=${RUN_ID}"
echo "SYNTH_NII=${SYNTH_NII}"
echo "SIF=${SIF}"
echo "LICENSE=${LICENSE}"

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

echo "== after freesurfer source =="
which recon-all
recon-all -version
which mri_synthsr

CUDA_VISIBLE_DEVICES="" mri_synthsr \
  --i "${INPUT_NII}" \
  --o "${SYNTH_NII}" \
  --threads "${SLURM_CPUS_PER_TASK}"

if [ ! -f "${SYNTH_NII}" ]; then
  echo "ERROR: SynthSR output not created: ${SYNTH_NII}"
  exit 1
fi

singularity exec --nv \
  --no-mount home \
  -B "${BASE_DIR}:/data" \
  -B "${TOP_DIR}/license.txt:/extra/license.txt" \
  "${SIF}" \
  /fastsurfer/run_fastsurfer.sh \
    --fs_license /extra/license.txt \
    --t1 "/data/Output/synthsr_output/${RUN_ID}_synthsr.nii.gz" \
    --sid "${RUN_ID}" \
    --sd /data/Output/segment_output \
    --threads "${SLURM_CPUS_PER_TASK}"

echo "Finished ${RUN_ID}"
