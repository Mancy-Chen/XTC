#!/bin/bash

# Preprocessing script to find FreeSurfer subjects for radiomics analysis
# Filters for subjects containing "base" or "long" in their names

# Input directory with FreeSurfer subjects
SUBJECTS_DIR="/data/projects/CSC/data/XTC/next_xtc/derivatives_unzipped"

# Output file for subject list
OUTPUT_FILE="radiomics_results/radiomics_subjects.txt"

# Name filters (must contain one of these strings)
NAME_FILTERS=("base" "long")

echo "Scanning for FreeSurfer subjects for radiomics analysis..."
echo "Subjects directory: ${SUBJECTS_DIR}"
echo "Filters: ${NAME_FILTERS[@]}"
echo ""

# Clear/create output file
> ${OUTPUT_FILE}

# Counters
TOTAL_SUBJECTS=0
MATCHED_SUBJECTS=0

# Scan directory
for subject_dir in ${SUBJECTS_DIR}/*/; do
    # Remove trailing slash
    subject_dir=${subject_dir%/}
    subject_name=$(basename ${subject_dir})
    
    # Check if mri directory exists
    if [ ! -d "${subject_dir}/mri" ]; then
        continue
    fi
    
    TOTAL_SUBJECTS=$((TOTAL_SUBJECTS + 1))
    
    # Check if subject name matches any filter
    matched=0
    for filter in "${NAME_FILTERS[@]}"; do
        if [[ "${subject_name}" == *"${filter}"* ]]; then
            matched=1
            break
        fi
    done
    
    if [ ${matched} -eq 1 ]; then
        echo "${subject_name}" >> ${OUTPUT_FILE}
        MATCHED_SUBJECTS=$((MATCHED_SUBJECTS + 1))
        echo "✓ ${subject_name}"
    fi
done

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "Total FreeSurfer subjects found: ${TOTAL_SUBJECTS}"
echo "Subjects matching filters: ${MATCHED_SUBJECTS}"
echo ""
echo "Subject list saved to: ${OUTPUT_FILE}"
echo ""
echo "Next step:"
echo "  sbatch --array=0-$((MATCHED_SUBJECTS - 1))%10 extract_radiomics_slurm.sh"
echo "=========================================="
