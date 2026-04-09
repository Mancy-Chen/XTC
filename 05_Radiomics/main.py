# Mancy Chen 08/07/2025
# This script extracts radiomics features (mean intensity and volume) from specific brain regions
import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor

# --- CONFIGURATION ---
#BASE_DIR = '/data/projects/CSC/data/XTC/next_xtc/derivatives_unzipped/'
BASE_DIR = '/data/projects/CSC/code/XTC/05_Radiomics/testing/'
label_list = [17, 18, 53, 54]
label_name = {
    17: 'Left-Hippocampus',
    18: 'Left-Amygdala',
    53: 'Right-Hippocampus',
    54: 'Right-Amygdala',
}

# Extractor for voxel-based volume
params = {
    'verbose': False,
    'firstorder': {'enable': True},
    'shape': {
        'enable': True,
        'VoxelVolume': True,
        'MeshVolume': False,
    },
    'glcm':   {'enable': False},
    'glrlm':  {'enable': False},
    'glszm':  {'enable': False},
    'ngtdm':  {'enable': False},
    'gldm':   {'enable': False}
}

extractor = featureextractor.RadiomicsFeatureExtractor(**params)

def load_mgz_as_sitk(path):
    img_nib = nib.load(path)
    data     = img_nib.get_fdata(dtype=np.float32)
    spacing  = [float(z) for z in img_nib.header.get_zooms()[:3]]
    origin   = [float(x) for x in img_nib.affine[:3, 3]]
    img_sitk = sitk.GetImageFromArray(data)
    img_sitk.SetSpacing(spacing)
    img_sitk.SetOrigin(origin)
    return img_sitk

# Resolve nested session folders
top_dirs = sorted(glob.glob(os.path.join(BASE_DIR, '*')))
session_dirs = []
for d in top_dirs:
    nested = os.path.join(d, os.path.basename(d))
    session_dirs.append(nested if os.path.isdir(nested) else d)

# Loop sessions and subjects
for sess_dir in session_dirs:
    dir_name = os.path.basename(sess_dir)
    print(dir_name)
    subj, sess, *_ = dir_name.split('_')
    out_csv       = "group_ROI_metrics.csv"

    try:
        t1_path   = os.path.join(sess_dir, 'mri', 'T1.mgz')
        mask_path = os.path.join(sess_dir, 'mri', 'aparc+aseg.mgz')
        img       = load_mgz_as_sitk(t1_path)
        atlas     = load_mgz_as_sitk(mask_path)

        # precompute voxel volume
        sx, sy, sz = img.GetSpacing()
        voxel_vol  = sx * sy * sz

        rows = []
        for lbl in label_list:
            bin_mask = sitk.BinaryThreshold(atlas, lbl, lbl, 1, 0)
            arr      = sitk.GetArrayFromImage(bin_mask)
            n_vox    = int(arr.sum())

            if n_vox == 0:
                vol_mm3 = np.nan
            elif n_vox == 1:
                vol_mm3 = voxel_vol
            else:
                feats   = extractor.execute(img, bin_mask)
                vol_key = next((k for k in feats if k.endswith('VoxelVolume')), None)
                vol_mm3 = feats[vol_key] if vol_key else (n_vox * voxel_vol)
            
            row = {
                'Subject':    subj,
                'Session':    sess,
                'LabelID':    lbl,
                'LabelName':  label_name[lbl],
                'Volume_mm3': vol_mm3
            }
            # Add all features to the row
            for feat_name, feat_value in feats.items():
                row[feat_name] = feat_value

            rows.append(row)
        # Collect all possible columns
        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())
        # Exclude columns starting with 'diagnostics'
        all_columns = [col for col in all_columns if not col.startswith('diagnostics')]
        all_columns = ['Subject', 'Session', 'LabelID', 'LabelName', 'Volume_mm3'] + \
                    [col for col in all_columns if col not in {'Subject', 'Session', 'LabelID', 'LabelName', 'Volume_mm3'}]

        df = pd.DataFrame(rows, columns=all_columns)
        # append if exists, else write new
        if os.path.exists(out_csv):
            df.to_csv(out_csv, mode='a', index=False, header=False)
        else:
            df.to_csv(out_csv, mode='w', index=False, header=True)

        print(f"Updated: {out_csv}")

    except Exception as e:
        print(f"Skipping {subj} {sess} due to error: {e}")

# You may now inspect the three session CSVs for complete and skipped entries.


#######################################################################################################################
# import os
# import glob
# import pandas as pd
# import matplotlib
# # matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt

# base_dir = '/data/projects/CSC/code/XTC/result/'
# pattern  = os.path.join(base_dir, '*_group_ROI_metrics.csv')
# files    = glob.glob(pattern)

# for fpath in files:
#     # load one session‐file
#     df = pd.read_csv(fpath)
#     session = os.path.basename(fpath).split('_')[0]  # e.g. 'sessie1'
#     df['Session'] = session

#     # pivot so each ROI is a column, one row per subject
#     p = (
#         df
#         .pivot(index='Subject', columns='LabelName', values='Volume_mm3')
#         .dropna(subset=['Left-Hippocampus','Right-Hippocampus','Left-Amygdala','Right-Amygdala'])
#     )

#     # Plot A: L vs R Hippocampus
#     plt.figure()
#     plt.scatter(p['Left-Hippocampus'], p['Right-Hippocampus'])
#     plt.xlabel('Left Hippocampus Volume (mm³)')
#     plt.ylabel('Right Hippocampus Volume (mm³)')
#     plt.title(f'{session}: Left vs Right Hippocampus')
#     plt.tight_layout()
#     plt.savefig(f'{session}_hipp_L_vs_R.png')
#     plt.show()

#     # Plot B: L Hippocampus vs L Amygdala
#     plt.figure()
#     plt.scatter(p['Left-Hippocampus'], p['Left-Amygdala'])
#     plt.xlabel('Left Hippocampus Volume (mm³)')
#     plt.ylabel('Left Amygdala Volume (mm³)')
#     plt.title(f'{session}: Hipp L vs Amyg L')
#     plt.tight_layout()
#     plt.savefig(f'{session}_hippL_vs_amygL.png')
#     plt.show()

#     # Plot C: R Hippocampus vs R Amygdala
#     plt.figure()
#     plt.scatter(p['Right-Hippocampus'], p['Right-Amygdala'])
#     plt.xlabel('Right Hippocampus Volume (mm³)')
#     plt.ylabel('Right Amygdala Volume (mm³)')
#     plt.title(f'{session}: Hipp R vs Amyg R')
#     plt.tight_layout()
#     plt.savefig(f'{session}_hippR_vs_amygR.png')
#     plt.show()


