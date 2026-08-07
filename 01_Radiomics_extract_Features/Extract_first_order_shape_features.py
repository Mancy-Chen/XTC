# Mancy Chen 2026-03-18
# use XTC interpreter
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from radiomics import featureextractor


IMAGE_ROOT = ".../XTC/01_Segmentation/fastsurfer-test/my_fastsurfer_analysis"
MASK_ROOT = ".../XTC/04_SynthSR/Whole_data/segment_output"
OUTPUT_DIR = ".../XTC/05_Radiomics_Mancy"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "radiomics_firstorder_shape_hippocampus_thalamus.csv")

TARGET_LABELS = {
    10: "Left_Thalamus",
    17: "Left_Hippocampus",
    49: "Right_Thalamus",
    53: "Right_Hippocampus",
}

MIN_VOXELS = 10


def mgz_to_sitk(mgz_path, force_int=False):
    nii = nib.load(mgz_path)
    data = nii.get_fdata()

    if force_int:
        data = np.rint(data).astype(np.int16)
    else:
        data = data.astype(np.float32)

    data_sitk = np.transpose(data, (2, 1, 0))
    img = sitk.GetImageFromArray(data_sitk)

    zooms = nii.header.get_zooms()[:3]
    img.SetSpacing((float(zooms[0]), float(zooms[1]), float(zooms[2])))

    affine = nii.affine
    origin = tuple(affine[:3, 3])

    direction_matrix = affine[:3, :3].copy()
    spacing = np.array(zooms, dtype=float)

    for i in range(3):
        if spacing[i] != 0:
            direction_matrix[:, i] /= spacing[i]

    img.SetOrigin(origin)
    img.SetDirection(tuple(direction_matrix.flatten(order="F")))

    return img


def resample_mask_to_image(mask_sitk, reference_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(mask_sitk)


def make_binary_mask(mask_sitk, label_value):
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    binary_arr = (mask_arr == label_value).astype(np.uint8)
    binary_mask = sitk.GetImageFromArray(binary_arr)
    binary_mask.CopyInformation(mask_sitk)
    return binary_mask


def voxel_count(mask_sitk):
    arr = sitk.GetArrayFromImage(mask_sitk)
    return int(np.sum(arr > 0))


def print_geometry_info(name, img):
    print(f"{name}:")
    print(f"  Size      = {img.GetSize()}")
    print(f"  Spacing   = {img.GetSpacing()}")
    print(f"  Origin    = {img.GetOrigin()}")
    print(f"  Direction = {img.GetDirection()}")


def get_subject_pairs():
    image_paths = glob.glob(os.path.join(IMAGE_ROOT, "*", "*", "mri", "orig_nu.mgz"))
    pairs = []

    for img_path in image_paths:
        subject_folder = img_path.split(os.sep)[-3]
        subject_base = subject_folder.replace("_3DT1", "")

        mask_path = os.path.join(
            MASK_ROOT,
            subject_base,
            "mri",
            "aparc.DKTatlas+aseg.deep.withCC.mgz"
        )

        if os.path.exists(mask_path):opt/amc/freesurfer-/
            pairs.append((subject_base, img_path, mask_path))
        else:
            print(f"[WARN] Missing mask for {subject_base}: {mask_path}")

    return pairs


os.makedirs(OUTPUT_DIR, exist_ok=True)

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("firstorder")
extractor.enableFeatureClassByName("shape")

extractor.settings["binWidth"] = 25
extractor.settings["normalize"] = False
extractor.settings["resampledPixelSpacing"] = None
extractor.settings["interpolator"] = sitk.sitkBSpline

# Optional: only use this if there are still tiny numerical mismatches after resampling
extractor.settings["geometryTolerance"] = 1e-3
extractor.settings["correctMask"] = False

all_results = []
subject_pairs = get_subject_pairs()

print(f"Found {len(subject_pairs)} matched subjects")

for subject_id, image_path, mask_path in subject_pairs:
    print(f"\nProcessing {subject_id}")

    try:
        image_sitk = mgz_to_sitk(image_path, force_int=False)
        mask_sitk_multi = mgz_to_sitk(mask_path, force_int=True)

        # debug before resampling
        print_geometry_info("Image", image_sitk)
        print_geometry_info("Mask_before", mask_sitk_multi)

        # align mask to image
        mask_sitk_multi = resample_mask_to_image(mask_sitk_multi, image_sitk)

        # debug after resampling
        print_geometry_info("Mask_after", mask_sitk_multi)

    except Exception as e:
        print(f"[ERROR] Failed loading {subject_id}: {e}")
        continue

    for label_value, roi_name in TARGET_LABELS.items():
        try:
            binary_mask = make_binary_mask(mask_sitk_multi, label_value)
            n_voxels = voxel_count(binary_mask)

            if n_voxels < MIN_VOXELS:
                print(f"  [SKIP] {roi_name} has only {n_voxels} voxels")
                continue

            result = extractor.execute(image_sitk, binary_mask)

            row = {
                "subject_id": subject_id,
                "roi_label": label_value,
                "roi_name": roi_name,
                "n_voxels": n_voxels,
                "image_path": image_path,
                "mask_path": mask_path,
            }

            for k, v in result.items():
                if k.startswith("original_firstorder_") or k.startswith("original_shape_"):
                    row[k] = v

            all_results.append(row)
            print(f"  [OK] {roi_name}")

        except Exception as e:
            print(f"  [ERROR] {subject_id} - {roi_name}: {e}")

df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_CSV, index=False)

print("\nDone.")
print(f"Saved to: {OUTPUT_CSV}")
print(f"Total rows: {len(df)}")
