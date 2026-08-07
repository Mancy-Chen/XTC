# Mancy Chen
# Extract PyRadiomics VoxelVolume for all brain regions
# Use XTC interpreter

import os
import glob
import re
import traceback

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk

from radiomics import featureextractor


# ============================================================
# PATHS AND SETTINGS
# ============================================================

IMAGE_ROOT = (
    ".../XTC/01_Segmentation/"
    "fastsurfer-test/my_fastsurfer_analysis"
)

MASK_ROOT = (
    ".../XTC/04_SynthSR/"
    "Whole_data/segment_output"
)

OUTPUT_DIR = (
    ".../XTC/05_Radiomics_Mancy"
)

OUTPUT_LONG_CSV = os.path.join(
    OUTPUT_DIR,
    "radiomics_voxelvolume_all_rois_long.csv"
)

OUTPUT_WIDE_CSV = os.path.join(
    OUTPUT_DIR,
    "radiomics_voxelvolume_all_rois_wide.csv"
)

OUTPUT_FAILED_CSV = os.path.join(
    OUTPUT_DIR,
    "radiomics_voxelvolume_all_rois_failed.csv"
)

MASK_FILENAME = "aparc.DKTatlas+aseg.deep.withCC.mgz"
IMAGE_FILENAME = "orig_nu.mgz"

BACKGROUND_LABEL = 0

# Retained from your previous analysis.
# Anatomical regions with fewer than 10 voxels will be skipped.
# Change this to 1 to attempt every nonzero label.
MIN_VOXELS = 10


# ============================================================
# FREESURFER LABEL LOOKUP TABLE
# ============================================================

def find_freesurfer_lut():
    """
    Find FreeSurferColorLUT.txt.

    Returns
    -------
    str or None
        Path to the lookup table.
    """
    candidates = []

    freesurfer_home = os.environ.get("FREESURFER_HOME")

    if freesurfer_home:
        candidates.append(
            os.path.join(
                freesurfer_home,
                "FreeSurferColorLUT.txt"
            )
        )

    candidates.extend(
        [
            (
                "/opt/amc/freesurfer-8.1.0/"
                "FreeSurferColorLUT.txt"
            ),
            (
                "/opt/amc/freesurfer-7.1.0/"
                "FreeSurferColorLUT.txt"
            ),
            (
                "/usr/local/freesurfer/"
                "FreeSurferColorLUT.txt"
            ),
        ]
    )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def load_freesurfer_lut(lut_path):
    """
    Read FreeSurferColorLUT.txt.

    Returns
    -------
    dict
        Mapping from numeric label ID to region name.
    """
    label_names = {}

    if lut_path is None:
        print(
            "[WARN] FreeSurferColorLUT.txt was not found. "
            "Numeric label names will be used."
        )
        return label_names

    with open(
        lut_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as lut_file:

        for line in lut_file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                label_value = int(parts[0])
            except ValueError:
                continue

            roi_name = parts[1]
            label_names[label_value] = roi_name

    return label_names


# ============================================================
# SUBJECT AND SESSION INFORMATION
# ============================================================

def parse_subject_session(subject_session):
    """
    Convert, for example:

        I010_sessie1

    into:

        subject_id = I010
        session = sessie1
    """
    match = re.match(
        r"^(.*?)_(sessie[^_]*)$",
        subject_session,
        flags=re.IGNORECASE
    )

    if match:
        subject_id = match.group(1)
        session = match.group(2)
    else:
        subject_id = subject_session
        session = "unknown"

    return subject_id, session


def safe_column_name(text):
    """
    Convert an ROI name into a safe CSV column name.
    """
    text = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(text)
    )

    return text.strip("_")


# ============================================================
# MGZ AND SIMPLEITK FUNCTIONS
# ============================================================

def mgz_mask_to_sitk(mgz_path):
    """
    Convert an MGZ segmentation to a SimpleITK label image.

    No resampling is performed.

    Nibabel array ordering:
        x, y, z

    SimpleITK array ordering:
        z, y, x
    """
    nii = nib.load(mgz_path)

    data = np.asanyarray(
        nii.dataobj
    )

    # Segmentation labels should be integers.
    rounded_data = np.rint(data)

    non_integer_voxels = int(
        np.count_nonzero(
            ~np.isclose(
                data,
                rounded_data,
                atol=1e-5
            )
        )
    )

    if non_integer_voxels > 0:
        print(
            f"  [WARN] Found {non_integer_voxels} "
            "non-integer segmentation voxels. "
            "They will be rounded."
        )

    data = rounded_data.astype(
        np.int32,
        copy=False
    )

    # Convert x-y-z to z-y-x.
    data_sitk = np.transpose(
        data,
        (2, 1, 0)
    )

    mask_sitk = sitk.GetImageFromArray(
        data_sitk
    )

    voxel_sizes = nib.affines.voxel_sizes(
        nii.affine
    )[:3]

    mask_sitk.SetSpacing(
        tuple(
            float(value)
            for value in voxel_sizes
        )
    )

    # Absolute origin and direction do not affect VoxelVolume.
    # The dummy image receives exactly the same geometry.
    mask_sitk.SetOrigin(
        (0.0, 0.0, 0.0)
    )

    mask_sitk.SetDirection(
        (
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    )

    return mask_sitk


def make_dummy_image(mask_sitk):
    """
    Create an intensity image with exactly the same geometry as
    the segmentation mask.

    MRI intensity values are not used for shape VoxelVolume.
    """
    dummy_image = sitk.Image(
        mask_sitk.GetSize(),
        sitk.sitkFloat32
    )

    dummy_image.CopyInformation(
        mask_sitk
    )

    return dummy_image


def make_binary_mask(mask_sitk, label_value):
    """
    Create a binary mask for one segmentation label.
    """
    mask_array = sitk.GetArrayFromImage(
        mask_sitk
    )

    binary_array = (
        mask_array == label_value
    ).astype(np.uint8)

    binary_mask = sitk.GetImageFromArray(
        binary_array
    )

    binary_mask.CopyInformation(
        mask_sitk
    )

    return binary_mask


def voxel_count(binary_mask):
    """
    Count nonzero voxels in a binary ROI mask.
    """
    mask_array = sitk.GetArrayViewFromImage(
        binary_mask
    )

    return int(
        np.count_nonzero(mask_array)
    )


# ============================================================
# SUBJECT–MASK PAIRING
# ============================================================

def get_subject_pairs():
    """
    Find every segmentation and construct its corresponding
    orig_nu.mgz path.

    The original image is retained for provenance, but its intensity
    values are not needed for VoxelVolume extraction.
    """
    mask_pattern = os.path.join(
        MASK_ROOT,
        "*",
        "mri",
        MASK_FILENAME
    )

    mask_paths = sorted(
        glob.glob(mask_pattern)
    )

    pairs = []

    for mask_path in mask_paths:

        # Example:
        # .../segment_output/I010_sessie1/mri/mask.mgz
        subject_session = os.path.basename(
            os.path.dirname(
                os.path.dirname(mask_path)
            )
        )

        image_folder = (
            f"{subject_session}_3DT1"
        )

        image_path = os.path.join(
            IMAGE_ROOT,
            image_folder,
            image_folder,
            "mri",
            IMAGE_FILENAME
        )

        if not os.path.exists(image_path):
            print(
                f"[WARN] Missing orig_nu.mgz for "
                f"{subject_session}:\n"
                f"       {image_path}\n"
                "       VoxelVolume extraction will continue "
                "because image intensities are not required."
            )

        pairs.append(
            (
                subject_session,
                image_path,
                mask_path
            )
        )

    return pairs


# ============================================================
# CONFIGURE PYRADIOMICS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

extractor = (
    featureextractor.RadiomicsFeatureExtractor(
        normalize=False,

        # Do not resample the mask.
        resampledPixelSpacing=None,

        # The dummy image and mask already match exactly.
        geometryTolerance=1e-6,

        # Do not let PyRadiomics alter the mask.
        correctMask=False,

        # Prevent large diagnostic outputs.
        additionalInfo=False,

        # Standard requirement for a 3D shape.
        minimumROIDimensions=2,

        # Voxel filtering is handled using MIN_VOXELS below.
        minimumROISize=None,
    )
)

# Disable every feature class.
extractor.disableAllFeatures()

# Enable only:
# original_shape_VoxelVolume
extractor.enableFeaturesByName(
    shape=["VoxelVolume"]
)

print(
    "Enabled PyRadiomics features:",
    extractor.enabledFeatures
)


# ============================================================
# LOAD ROI NAMES
# ============================================================

lut_path = find_freesurfer_lut()
label_name_lookup = load_freesurfer_lut(
    lut_path
)

if lut_path is not None:
    print(
        f"Using FreeSurfer LUT:\n{lut_path}"
    )


# ============================================================
# BATCH EXTRACTION
# ============================================================

all_results = []
failed_results = []

subject_pairs = get_subject_pairs()

print(
    f"\nFound {len(subject_pairs)} "
    "segmentation masks"
)

for case_index, (
    subject_session,
    image_path,
    mask_path
) in enumerate(
    subject_pairs,
    start=1
):

    subject_id, session = (
        parse_subject_session(
            subject_session
        )
    )

    print()
    print(
        f"[{case_index}/{len(subject_pairs)}] "
        f"Processing {subject_session}"
    )

    try:
        # Load the multi-label segmentation.
        mask_sitk_multi = mgz_mask_to_sitk(
            mask_path
        )

        # Dummy image matches the segmentation exactly.
        image_sitk = make_dummy_image(
            mask_sitk_multi
        )

        mask_array = sitk.GetArrayViewFromImage(
            mask_sitk_multi
        )

        labels, counts = np.unique(
            mask_array,
            return_counts=True
        )

        # Remove background label 0.
        valid = labels != BACKGROUND_LABEL

        labels = labels[valid]
        counts = counts[valid]

        label_count_lookup = {
            int(label): int(count)
            for label, count in zip(
                labels,
                counts
            )
        }

        voxel_spacing = np.asarray(
            mask_sitk_multi.GetSpacing(),
            dtype=float
        )

        single_voxel_volume_mm3 = float(
            np.prod(voxel_spacing)
        )

        print(
            f"  Found {len(labels)} "
            "nonzero labels"
        )

    except Exception as error:
        print(
            f"  [ERROR] Failed loading "
            f"{subject_session}: {error}"
        )

        failed_results.append(
            {
                "subject_session": subject_session,
                "subject_id": subject_id,
                "session": session,
                "roi_label": None,
                "roi_name": None,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )

        continue

    successful_rois = 0
    skipped_rois = 0
    failed_rois = 0

    # Loop through every nonzero segmentation label.
    for label_value in labels:

        label_value = int(label_value)

        roi_name = label_name_lookup.get(
            label_value,
            f"Unknown_Label_{label_value}"
        )

        n_voxels = label_count_lookup[
            label_value
        ]

        if n_voxels < MIN_VOXELS:
            print(
                f"  [SKIP] Label {label_value} "
                f"({roi_name}) has only "
                f"{n_voxels} voxels"
            )

            skipped_rois += 1
            continue

        try:
            binary_mask = make_binary_mask(
                mask_sitk_multi,
                label_value
            )

            result = extractor.execute(
                image_sitk,
                binary_mask,
                label=1,
                voxelBased=False
            )

            feature_name = (
                "original_shape_VoxelVolume"
            )

            if feature_name not in result:
                raise KeyError(
                    f"{feature_name} was not returned. "
                    f"Returned features: "
                    f"{list(result.keys())}"
                )

            pyradiomics_volume_mm3 = float(
                result[feature_name]
            )

            # Independent manual QC calculation.
            manual_volume_mm3 = float(
                n_voxels
                * single_voxel_volume_mm3
            )

            difference_mm3 = float(
                pyradiomics_volume_mm3
                - manual_volume_mm3
            )

            column_name = (
                f"{label_value}_"
                f"{safe_column_name(roi_name)}_"
                "VoxelVolume_mm3"
            )

            row = {
                "subject_session": subject_session,
                "subject_id": subject_id,
                "session": session,
                "roi_label": label_value,
                "roi_name": roi_name,
                "n_voxels": n_voxels,
                "voxel_size_x_mm": float(
                    voxel_spacing[0]
                ),
                "voxel_size_y_mm": float(
                    voxel_spacing[1]
                ),
                "voxel_size_z_mm": float(
                    voxel_spacing[2]
                ),
                "single_voxel_volume_mm3": (
                    single_voxel_volume_mm3
                ),
                "original_shape_VoxelVolume": (
                    pyradiomics_volume_mm3
                ),
                "voxel_volume_ml": (
                    pyradiomics_volume_mm3
                    / 1000.0
                ),
                "manual_volume_mm3": (
                    manual_volume_mm3
                ),
                "difference_mm3": (
                    difference_mm3
                ),
                "wide_column_name": column_name,
                "image_path": image_path,
                "mask_path": mask_path,
            }

            all_results.append(row)
            successful_rois += 1

            print(
                f"  [OK] {label_value}: "
                f"{roi_name} = "
                f"{pyradiomics_volume_mm3:.3f} mm³"
            )

        except Exception as error:
            failed_rois += 1

            print(
                f"  [ERROR] {subject_session} - "
                f"{label_value} ({roi_name}): "
                f"{error}"
            )

            failed_results.append(
                {
                    "subject_session": subject_session,
                    "subject_id": subject_id,
                    "session": session,
                    "roi_label": label_value,
                    "roi_name": roi_name,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )

    print(
        f"  Summary: {successful_rois} extracted, "
        f"{skipped_rois} skipped, "
        f"{failed_rois} failed"
    )


# ============================================================
# SAVE LONG-FORM OUTPUT
# ============================================================

df_long = pd.DataFrame(
    all_results
)

if df_long.empty:
    raise RuntimeError(
        "No ROI volumes were extracted."
    )

df_long = df_long.sort_values(
    [
        "subject_id",
        "session",
        "roi_label",
    ]
).reset_index(drop=True)

df_long.to_csv(
    OUTPUT_LONG_CSV,
    index=False
)


# ============================================================
# SAVE WIDE-FORM OUTPUT
# ============================================================

df_wide = df_long.pivot_table(
    index=[
        "subject_session",
        "subject_id",
        "session",
    ],
    columns="wide_column_name",
    values="original_shape_VoxelVolume",
    aggfunc="first"
).reset_index()

df_wide.columns.name = None

metadata_columns = [
    "subject_session",
    "subject_id",
    "session",
]

roi_columns = [
    column
    for column in df_wide.columns
    if column not in metadata_columns
]


def get_label_from_column(column_name):
    """
    Obtain the numeric label at the beginning of a wide column.
    """
    match = re.match(
        r"^(\d+)_",
        str(column_name)
    )

    if match:
        return int(match.group(1))

    return 10**9


roi_columns = sorted(
    roi_columns,
    key=lambda column: (
        get_label_from_column(column),
        str(column)
    )
)

df_wide = df_wide[
    metadata_columns + roi_columns
]

df_wide = df_wide.sort_values(
    [
        "subject_id",
        "session",
    ]
).reset_index(drop=True)

# Missing regions remain NaN rather than being changed to zero.
# A missing segmentation label may represent a QC problem.
df_wide.to_csv(
    OUTPUT_WIDE_CSV,
    index=False
)


# ============================================================
# SAVE FAILURES
# ============================================================

df_failed = pd.DataFrame(
    failed_results
)

if not df_failed.empty:
    df_failed.to_csv(
        OUTPUT_FAILED_CSV,
        index=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("Done")
print("=" * 70)

print(
    f"Long-format output:\n"
    f"{OUTPUT_LONG_CSV}"
)

print(
    f"\nWide-format output:\n"
    f"{OUTPUT_WIDE_CSV}"
)

if not df_failed.empty:
    print(
        f"\nFailed extraction output:\n"
        f"{OUTPUT_FAILED_CSV}"
    )

print(
    f"\nSubjects/sessions: "
    f"{df_long['subject_session'].nunique()}"
)

print(
    f"Unique ROI labels: "
    f"{df_long['roi_label'].nunique()}"
)

print(
    f"Total ROI rows: "
    f"{len(df_long)}"
)

print(
    "\nMaximum absolute difference between "
    "PyRadiomics and manual volume:"
)

print(
    f"{df_long['difference_mm3'].abs().max():.12g} mm³"
)
