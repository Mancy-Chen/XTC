"""
FreeSurfer Radiomics Feature Extraction
Extracts radiomics features from FreeSurfer processed subjects using PyRadiomics
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from radiomics import featureextractor
import SimpleITK as sitk

# ============================================================================
# CONFIGURATION
# ============================================================================

# We really only want base and long results analyzes
NAME_FILTERS = [
    "base",
    "long",
]

# PyRadiomics feature classes - toggle on/off as needed
FEATURE_CLASSES = {
    'firstorder': True,      # First Order Statistics
    'glcm': True,            # Gray Level Co-occurrence Matrix
    'glrlm': True,           # Gray Level Run Length Matrix
    'glszm': True,           # Gray Level Size Zone Matrix
    'gldm': True,            # Gray Level Dependence Matrix
    'ngtdm': True,           # Neighbouring Gray Tone Difference Matrix
    'shape': True,           # Shape-based features
}

# PyRadiomics settings
RADIOMICS_SETTINGS = {
    'binWidth': 25,                    # Histogram bin width
    'resampledPixelSpacing': None,     # None = no resampling, or [1,1,1] for isotropic
    'interpolator': sitk.sitkBSpline,  # Interpolation method
    'verbose': False,
    'normalize': False,                # Normalize image before extraction
    'normalizeScale': 1,
    'removeOutliers': None,
}

# FreeSurfer volumes to extract features from
# Format: (filename, label_value_or_None, description)
FREESURFER_VOLUMES = [
    ('norm.mgz', None, 'whole_brain'),
    #('norm.mgz', [2, 41], 'white_matter'),         # Left + Right cerebral white matter
    #('norm.mgz', [3, 42], 'cortex'),               # Left + Right cerebral cortex
    ('norm.mgz', [17], 'left_hippocampus'),
    ('norm.mgz', [53], 'right_hippocampus'),
    ('norm.mgz', [18], 'left_amygdala'),
    ('norm.mgz', [54], 'right_amygdala'),
    ('norm.mgz', [10, 49], 'thalamus'),            # Left + Right thalamus
    ('norm.mgz', [11, 50], 'caudate'),             # Left + Right caudate
    ('norm.mgz', [12, 51], 'putamen'),             # Left + Right putamen
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_logging(output_dir, subject_id):
    """Setup logging for the extraction process"""
    log_file = output_dir / f"{subject_id}_radiomics.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_mask_from_labels(aseg_path, label_values):
    """
    Create a binary mask from specific label values in aseg.mgz
    
    Args:
        aseg_path: Path to aseg.mgz file
        label_values: List of label values to include in mask, or None for whole volume
    
    Returns:
        SimpleITK image with binary mask
    """
    # Load aseg volume
    aseg_nib = nib.load(aseg_path)
    aseg_data = aseg_nib.get_fdata()
    
    if label_values is None:
        # Use entire volume (non-zero voxels)
        mask_data = (aseg_data > 0).astype(np.uint8)
    else:
        # Create mask from specific labels
        mask_data = np.zeros_like(aseg_data, dtype=np.uint8)
        for label in label_values:
            mask_data[aseg_data == label] = 1
    
    # Convert to SimpleITK
    mask_img = sitk.GetImageFromArray(mask_data.transpose(2, 1, 0))
    
    # Copy geometry from original using nibabel
    aseg_sitk = load_mgz_as_sitk(str(aseg_path))
    mask_img.CopyInformation(aseg_sitk)
    
    return mask_img


def load_mgz_as_sitk(mgz_path):
    """
    Load FreeSurfer .mgz file and convert to SimpleITK image
    
    Args:
        mgz_path: Path to .mgz file
    
    Returns:
        SimpleITK image
    """
    # Load with nibabel
    nib_img = nib.load(mgz_path)
    data = nib_img.get_fdata()
    
    # Convert to SimpleITK (transpose for correct orientation)
    sitk_img = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    
    # Set spacing from nibabel header
    # Use pixdim (voxel dimensions) which is the proper spacing
    spacing = nib_img.header.get_zooms()[:3]
    sitk_img.SetSpacing([float(s) for s in spacing])
    
    # Set origin from affine
    affine = nib_img.affine
    origin = affine[:3, 3]
    sitk_img.SetOrigin(origin.tolist())
    
    # Set direction from affine (normalized by spacing)
    direction_matrix = affine[:3, :3].copy()
    for i in range(3):
        direction_matrix[:, i] = direction_matrix[:, i] / spacing[i]
    sitk_img.SetDirection(direction_matrix.flatten().tolist())
    
    return sitk_img


def extract_features_for_volume(extractor, image_path, mask, roi_name, logger):
    """
    Extract radiomics features for a specific ROI
    
    Args:
        extractor: PyRadiomics feature extractor
        image_path: Path to the image file
        mask: SimpleITK mask image
        roi_name: Name of the ROI
        logger: Logger instance
    
    Returns:
        Dictionary of extracted features
    """
    logger.info(f"Extracting features for {roi_name}...")
    
    try:
        # Load image (handle .mgz format)
        image = load_mgz_as_sitk(str(image_path))
        
        # Extract features
        features = extractor.execute(image, mask)
        
        # Filter out diagnostic features (start with 'diagnostics_')
        feature_dict = {
            key: float(value) for key, value in features.items()
            if not key.startswith('diagnostics_')
        }
        
        logger.info(f"Extracted {len(feature_dict)} features for {roi_name}")
        return feature_dict
        
    except Exception as e:
        logger.error(f"Failed to extract features for {roi_name}: {str(e)}")
        return {}


def process_subject(subject_dir, output_dir, logger):
    """
    Process a single FreeSurfer subject
    
    Args:
        subject_dir: Path to subject's FreeSurfer output directory
        output_dir: Path to output directory for results
        logger: Logger instance
    
    Returns:
        Dictionary of all extracted features
    """
    subject_id = subject_dir.name
    logger.info(f"Processing subject: {subject_id}")
    
    # Check if mri directory exists
    mri_dir = subject_dir / 'mri'
    if not mri_dir.exists():
        logger.error(f"MRI directory not found: {mri_dir}")
        return None
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_SETTINGS)
    
    # Enable/disable feature classes based on configuration
    for feature_class, enabled in FEATURE_CLASSES.items():
        if enabled:
            extractor.enableFeatureClassByName(feature_class)
        else:
            extractor.disableFeatureClassByName(feature_class)
    
    # Dictionary to store all features
    all_features = {
        'subject_id': subject_id,
        'rois': {}
    }
    
    # Process each ROI
    for volume_file, labels, roi_name in FREESURFER_VOLUMES:
        volume_path = mri_dir / volume_file
        
        if not volume_path.exists():
            logger.warning(f"Volume not found: {volume_path}, skipping {roi_name}")
            continue
        
        # Create mask
        if labels is None and volume_file == 'nu.mgz':
            # For nu.mgz whole brain, create mask from brain.mgz or brainmask
            brainmask_path = mri_dir / 'brainmask.mgz'
            if brainmask_path.exists():
                mask = load_mgz_as_sitk(str(brainmask_path))
                mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
            else:
                # Fallback: use nu.mgz itself with threshold
                mask = load_mgz_as_sitk(str(volume_path))
                mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        else:
            # Create mask from aseg labels
            aseg_path = mri_dir / 'aseg.mgz'
            if not aseg_path.exists():
                logger.warning(f"aseg.mgz not found, skipping {roi_name}")
                continue
            mask = create_mask_from_labels(aseg_path, labels)
        
        # Check if mask is empty
        mask_array = sitk.GetArrayFromImage(mask)
        if mask_array.sum() == 0:
            logger.warning(f"Empty mask for {roi_name}, skipping")
            continue
        
        # Extract features
        features = extract_features_for_volume(
            extractor, volume_path, mask, roi_name, logger
        )
        
        if features:
            all_features['rois'][roi_name] = features
    
    # Save results
    output_file = output_dir / f"{subject_id}_radiomics.json"
    with open(output_file, 'w') as f:
        json.dump(all_features, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    return all_features


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract radiomics features from FreeSurfer subjects'
    )
    parser.add_argument(
        'subjects_dir',
        type=str,
        help='Directory containing FreeSurfer subject directories'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='radiomics_results',
        help='Output directory for results (default: radiomics_results)'
    )
    parser.add_argument(
        '--subject',
        type=str,
        help='Process only this specific subject (default: process all)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    subjects_dir = Path(args.subjects_dir)
    output_dir = Path(args.output_dir)
    
    if not subjects_dir.exists():
        print(f"ERROR: Subjects directory not found: {subjects_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get list of subjects to process
    if args.subject:
        subject_dirs = [subjects_dir / args.subject]
    else:
        # Find all directories that contain an 'mri' subdirectory
        # and match the name filters
        subject_dirs = [d for d in subjects_dir.iterdir() 
                       if d.is_dir() and (d / 'mri').exists() and
                       any(filter_str in d.name for filter_str in NAME_FILTERS)]
    
    if not subject_dirs:
        print("ERROR: No valid FreeSurfer subjects found")
        sys.exit(1)
    
    print(f"Found {len(subject_dirs)} subject(s) to process")
    print(f"Output directory: {output_dir}")
    print("\nEnabled feature classes:")
    for feature_class, enabled in FEATURE_CLASSES.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature_class}")
    print()
    
    # Process subjects
    for subject_dir in subject_dirs:

        logger = setup_logging(output_dir, subject_dir.name)
        try:
            process_subject(subject_dir, output_dir, logger)
            print(f"✓ Completed: {subject_dir.name}")
        except Exception as e:
            print(f"✗ Failed: {subject_dir.name} - {str(e)}")
            logger.exception("Processing failed")
    
    print(f"\nProcessing complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
