# Create bid structure for MRIQC input from T1-weighted images
import os
import shutil
from pathlib import Path

# INPUT: root folder where your T1s are stored
input_root = Path("/data/projects/CSC/data/2009-next-mmdewin/2019-next-mmdewin/data_maartje/MR_ruwedata_nifti/3DT1_from_GE_files_fixed_reorientation")

# OUTPUT: where to organize into BIDS-like format
output_root = Path("/data/projects/CSC/code/XTC/Quality_check/mriqc_input_bids")

# Create the output root if it doesn't exist
output_root.mkdir(parents=True, exist_ok=True)

for subj_dir in input_root.iterdir():
    if subj_dir.is_dir():
        subject = subj_dir.name  # e.g., I010
        for file in subj_dir.glob("*.nii.gz"):
            if "sessie" in file.name.lower():
                session = file.name.split("_")[0]  # e.g., sessie1
                bids_sub = f"sub-{subject}"
                bids_ses = f"ses-{session}"
                out_dir = output_root / bids_sub / bids_ses / "anat"
                out_dir.mkdir(parents=True, exist_ok=True)

                # BIDS-style filename
                bids_filename = f"{bids_sub}_{bids_ses}_T1w.nii.gz"
                shutil.copy(file, out_dir / bids_filename)
                print(f"Copied {file} → {out_dir / bids_filename}")

#######################################################################################################################
# Rename the ses part of the filename to match BIDS standards
import os
from pathlib import Path
import re

# Set the path to your BIDS input directory
bids_root = Path("/data/projects/CSC/code/XTC/Quality_check/mriqc_input_bids")

# Loop over all subjects
for subj_dir in bids_root.glob("sub-*"):
    if subj_dir.is_dir():
        for ses_dir in subj_dir.glob("ses-sessie*"):
            match = re.match(r"ses-sessie(\d+)", ses_dir.name)
            if match:
                ses_num = match.group(1)
                new_name = f"ses-{ses_num}"
                new_path = subj_dir / new_name
                print(f"Renaming {ses_dir} → {new_path}")
                ses_dir.rename(new_path)

                # Also rename the .nii.gz file inside `anat/`
                anat_dir = new_path / "anat"
                for file in anat_dir.glob("*sessie*.nii.gz"):
                    new_fname = file.name.replace(f"sessie{ses_num}", ses_num)
                    new_fpath = anat_dir / new_fname
                    print(f"Renaming {file.name} → {new_fname}")
                    file.rename(new_fpath)
