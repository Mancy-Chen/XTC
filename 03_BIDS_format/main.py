# Mancy Chen 16/07/2025
##############################################################################################################
# Dicom to nii
import os
import subprocess

# Paths
dcm2niix_path = "/opt/amc/mricron-1.0.20190902/dcm2niix"
src_dir = "/data/projects/CSC/data/2009-next-mmdewin/2019-next-mmdewin/data_maartje/MR_ruwedata_dicom/I010/TP1_20020410/00009_DWI"
dst_dir = "/data/projects/CSC/data/XTC/Dicom2nii_bids/sub-I010/ses-1/dwi"

# Ensure output directory exists
os.makedirs(dst_dir, exist_ok=True)

# Build the command
cmd = [
    dcm2niix_path,
    "-z", "y",                                   # compress output to .nii.gz
    "-f", "sub-I010_ses-1_dwi%p%t%s",             # filename template
    "-o", dst_dir,                               # output directory
    src_dir                                      # input DICOM directory
]

# Run conversion
try:
    subprocess.run(cmd, check=True)
    print(f"Conversion complete! NIfTI files are in:\n  {dst_dir}")
except subprocess.CalledProcessError as e:
    print("Error during conversion:", e)



##############################################################################################################
# This script put data to BIDS format
import pydicom
import os

# Example folder — update this if needed
dicom_folder = "/data/projects/CSC/data/2009-next-mmdewin/2019-next-mmdewin/data_maartje/MR_ruwedata_GE_format/I010/sessie1/3D/3D/"
sample_file = sorted(os.listdir(dicom_folder))[0]
sample_path = os.path.join(dicom_folder, sample_file)

# Read DICOM header
ds = pydicom.dcmread(sample_path, stop_before_pixels=True, force=True)

# Print useful fields
fields = [
    "PatientID", "StudyDate", "SeriesDescription", "ProtocolName",
    "SeriesNumber", "Modality", "ImageType"
]

print(f"\nInspecting DICOM file: {sample_file}\n")
for field in fields:
    try:
        value = ds.get(field, None)
        if value:
            print(f"{field}: {value}")
    except Exception as e:
        print(f"{field}: [error reading] {e}")


##################################################################################
#!/usr/bin/env python3
import os
import re
import sys
import zipfile
import subprocess

# === CONFIGURATION ===
ROOT_DICOM = "/data/projects/CSC/data/2009-next-mmdewin/2019-next-mmdewin/data_maartje/MR_ruwedata_dicom"
BIDS_ROOT  = "/data/projects/CSC/data/XTC/Dicom2nii_bids"
DCM2NIIX   = "/opt/amc/mricron-1.0.20190902/dcm2niix"

# Only handle these modality patterns → (BIDS folder, filename suffix)
MOD_PATTERNS = {
    re.compile(r"(DTI|DWI)", re.I):    ("dwi",  "dwi"),
    re.compile(r"T1w_3D_FSPGR", re.I): ("anat", "T1w")
}

TP_DIR_RE = re.compile(r"TP(\d+)_")  # to sort session folders by that number

def find_subject_dirs(root):
    """Yield full paths of subdirectories named I<digits> under root."""
    for entry in sorted(os.listdir(root)):
        if re.fullmatch(r"I\d+", entry):
            yield os.path.join(root, entry)

def find_session_dirs(subj_dir):
    """Return a list of (session_index, full_path) tuples, sorted by TP#."""
    candidates = []
    for entry in os.listdir(subj_dir):
        m = TP_DIR_RE.match(entry)
        path = os.path.join(subj_dir, entry)
        if m and os.path.isdir(path):
            tp_num = int(m.group(1))
            candidates.append((tp_num, path))
    # sort by TP number, then assign 1-based ses index
    candidates.sort(key=lambda x: x[0])
    return [(i+1, path) for i, (_, path) in enumerate(candidates)]

def match_modality(fname):
    """Return (mod_folder, fname_suffix) if ZIP name matches, else (None,None)."""
    for patt, out in MOD_PATTERNS.items():
        if patt.search(fname):
            return out
    return (None, None)

def unzip_if_needed(zip_path, extract_dir):
    if not os.path.isdir(extract_dir):
        print(f"      • Extracting {os.path.basename(zip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

def convert_series(subj, ses_idx, mod, suffix, src_dir):
    out_dir = os.path.join(BIDS_ROOT, f"sub-{subj}", f"ses-{ses_idx}", mod)
    os.makedirs(out_dir, exist_ok=True)
    tpl = f"sub-{subj}_ses-{ses_idx}_{suffix}"
    cmd = [
        DCM2NIIX,
        "-z", "y",
        "-f", tpl,
        "-o", out_dir,
        src_dir
    ]
    print(f"      • Converting → sub-{subj}/ses-{ses_idx}/{mod}")
    subprocess.run(cmd, check=True)

def process_subject(subj_dir):
    subj = os.path.basename(subj_dir)
    sessions = find_session_dirs(subj_dir)
    if not sessions:
        print(f"Skipping {subj}: no TP* folders found.", file=sys.stderr)
        return
    for ses_idx, ses_path in sessions:
        print(f"\n▶ Processing sub-{subj} ses-{ses_idx} ({os.path.basename(ses_path)})")
        for fname in os.listdir(ses_path):
            if not fname.lower().endswith(".zip"):
                continue
            mod, suffix = match_modality(fname)
            if not mod:
                continue
            zip_path   = os.path.join(ses_path, fname)
            extract_to = os.path.join(ses_path, os.path.splitext(fname)[0])
            try:
                unzip_if_needed(zip_path, extract_to)
                convert_series(subj, ses_idx, mod, suffix, extract_to)
            except Exception as e:
                print(f"    !!! Error on {zip_path}: {e}", file=sys.stderr)
                sys.exit(1)

def main():
    for subj_dir in find_subject_dirs(ROOT_DICOM):
        process_subject(subj_dir)
    print("\n✅ All done for all subjects & 3 sessions each.")

if __name__ == "__main__":
    main()
##############################################################################################################
# Convert .sav to .csv
import pyreadstat

df, meta = pyreadstat.read_sav("/data/projects/CSC/data/2009-next-mmdewin/2019-next-mmdewin/SPSS databases/database_cohortIII_update.sav")
print(df.head())
print(meta.column_names)   # variable names
print(meta.variable_value_labels)  # value labels
df.to_csv("/data/projects/CSC/data/2009-next-mmdewin/2019-next-mmdewin/SPSS databases/database_cohortIII_update.csv", index=False)



