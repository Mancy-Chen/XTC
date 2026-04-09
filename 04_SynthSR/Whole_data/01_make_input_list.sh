#!/bin/bash
set -euo pipefail

RAW_DIR="/data/projects/CSC/data/2009-next-mmdewin/2019-next-mmdewin/data_maartje/MR_ruwedata_nifti/3DT1_from_GE_files_fixed_reorientation"
OUT_LIST="/data/projects/CSC/code/XTC/04_SynthSR/test/segmentation/input_list.txt"

mkdir -p "$(dirname "$OUT_LIST")"

find "$RAW_DIR" -mindepth 2 -maxdepth 2 -type f -name 'sessie*_3DT1.nii.gz' | sort > "$OUT_LIST"

echo "Wrote list to: $OUT_LIST"
wc -l "$OUT_LIST"
head "$OUT_LIST"
