#!/bin/bash
#SBATCH --job-name=mriqc
#SBATCH --output=/data/projects/CSC/code/XTC/02_Quality_check/logs/mriqc_%A_%a.out
#SBATCH --error=/data/projects/CSC/code/XTC/02_Quality_check/logs/mriqc_%A_%a.err
#SBATCH --array=1-423  # <-- adjust this based on number of subjects
#SBATCH --nice=11
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ==== CONFIGURATION ====
BIDS_DIR=/data/projects/CSC/code/XTC/02_Quality_check/mriqc_input_bids
OUTPUT_DIR=/data/projects/CSC/code/XTC/02_Quality_check/mriqc_output
SINGULARITY_IMAGE=/scratch/mchen/mriqc_working_env/mriqc_24.0.2.sif
FS_LICENSE=/scratch/mchen/mriqc_working_env/license.txt
SUBJECTS=($(ls $BIDS_DIR | grep '^sub-'))

# Get the subject for this array task
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "Running MRIQC for $SUBJECT"

singularity run --cleanenv \
    -B ${BIDS_DIR}:/data \
    -B ${OUTPUT_DIR}:/out \
    -B ${FS_LICENSE}:/fs/license.txt \
    $SINGULARITY_IMAGE \
    /data /out participant \
    --participant-label ${SUBJECT#sub-} \
    --n_cpus 4 \
    --mem_gb 12 \

