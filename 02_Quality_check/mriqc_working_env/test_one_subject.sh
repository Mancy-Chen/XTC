#!/bin/bash

# ======= CONFIGURATION =======
SUBJECT_ID="sub-I078"

BIDS_DIR=/data/projects/CSC/code/XTC/02_Quality_check/mriqc_input_bids
OUTPUT_DIR=/data/projects/CSC/code/XTC/02_Quality_check/mriqc_output
SINGULARITY_IMAGE=/scratch/mchen/mriqc_working_env/mriqc_24.0.2.sif
FS_LICENSE=/scratch/mchen/mriqc_working_env/license.txt

# ======= EXECUTION ===========
echo "Running MRIQC for ${SUBJECT_ID}"

singularity run --cleanenv \
    -B ${BIDS_DIR}:/data \
    -B ${OUTPUT_DIR}:/out \
    -B ${FS_LICENSE}:/fs/license.txt \
    ${SINGULARITY_IMAGE} \
    /data /out participant \
    --participant-label I010 \
    --n_cpus 4 \
    --mem_gb 12 \


