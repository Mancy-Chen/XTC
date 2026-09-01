"""Project loadings using an optional external atlas set by XTC_SPATIAL_ATLAS."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    PCA_WHOLE_BRAIN_OUT,
    PLOT_SPATIAL_PROJECTION_OUT,
    SPATIAL_ATLAS_PATH,
    SPATIAL_PROJECTION_OUT,
)
from utils import ensure_dirs


SUBCORTICAL_LABELS = {
    "left_cerebral_white_matter": (2, "Left-Cerebral-White-Matter"),
    "left_lateral_ventricle": (4, "Left-Lateral-Ventricle"),
    "left_inf_lat_vent": (5, "Left-Inf-Lat-Vent"),
    "left_cerebellum_white_matter": (7, "Left-Cerebellum-White-Matter"),
    "left_cerebellum_cortex": (8, "Left-Cerebellum-Cortex"),
    "left_thalamus": (10, "Left-Thalamus"),
    "left_caudate": (11, "Left-Caudate"),
    "left_putamen": (12, "Left-Putamen"),
    "left_pallidum": (13, "Left-Pallidum"),
    "3rd_ventricle": (14, "3rd-Ventricle"),
    "4th_ventricle": (15, "4th-Ventricle"),
    "brain_stem": (16, "Brain-Stem"),
    "left_hippocampus": (17, "Left-Hippocampus"),
    "left_amygdala": (18, "Left-Amygdala"),
    "csf": (24, "CSF"),
    "left_accumbens_area": (26, "Left-Accumbens-area"),
    "left_ventraldc": (28, "Left-VentralDC"),
    "left_choroid_plexus": (31, "Left-choroid-plexus"),
    "right_cerebral_white_matter": (41, "Right-Cerebral-White-Matter"),
    "right_lateral_ventricle": (43, "Right-Lateral-Ventricle"),
    "right_inf_lat_vent": (44, "Right-Inf-Lat-Vent"),
    "right_cerebellum_white_matter": (46, "Right-Cerebellum-White-Matter"),
    "right_cerebellum_cortex": (47, "Right-Cerebellum-Cortex"),
    "right_thalamus": (49, "Right-Thalamus"),
    "right_caudate": (50, "Right-Caudate"),
    "right_putamen": (51, "Right-Putamen"),
    "right_pallidum": (52, "Right-Pallidum"),
    "right_hippocampus": (53, "Right-Hippocampus"),
    "right_amygdala": (54, "Right-Amygdala"),
    "right_accumbens_area": (58, "Right-Accumbens-area"),
    "right_ventraldc": (60, "Right-VentralDC"),
    "right_choroid_plexus": (63, "Right-choroid-plexus"),
    "wm_hypointensities": (77, "WM-hypointensities"),
    "cc_posterior": (251, "CC_Posterior"),
    "cc_mid_posterior": (252, "CC_Mid_Posterior"),
    "cc_central": (253, "CC_Central"),
    "cc_mid_anterior": (254, "CC_Mid_Anterior"),
    "cc_anterior": (255, "CC_Anterior"),
}

CORTICAL_IDS = {
    "caudalanteriorcingulate": 2,
    "caudalmiddlefrontal": 3,
    "cuneus": 5,
    "entorhinal": 6,
    "fusiform": 7,
    "inferiorparietal": 8,
    "inferiortemporal": 9,
    "isthmuscingulate": 10,
    "lateraloccipital": 11,
    "lateralorbitofrontal": 12,
    "lingual": 13,
    "medialorbitofrontal": 14,
    "middletemporal": 15,
    "parahippocampal": 16,
    "paracentral": 17,
    "parsopercularis": 18,
    "parsorbitalis": 19,
    "parstriangularis": 20,
    "pericalcarine": 21,
    "postcentral": 22,
    "posteriorcingulate": 23,
    "precentral": 24,
    "precuneus": 25,
    "rostralanteriorcingulate": 26,
    "rostralmiddlefrontal": 27,
    "superiorfrontal": 28,
    "superiorparietal": 29,
    "superiortemporal": 30,
    "supramarginal": 31,
    "transversetemporal": 34,
    "insula": 35,
}


def build_label_mapping() -> dict[str, tuple[int, str]]:
    mapping = dict(SUBCORTICAL_LABELS)
    for region, suffix in CORTICAL_IDS.items():
        mapping[f"ctx_lh_{region}"] = (1000 + suffix, f"ctx-lh-{region}")
        mapping[f"ctx_rh_{region}"] = (2000 + suffix, f"ctx-rh-{region}")
    return mapping


def slice_indices(length: int, n: int = 5) -> np.ndarray:
    if length <= n:
        return np.arange(length)
    return np.linspace(int(length * 0.15), int(length * 0.85), n).round().astype(int)


def crop_to_mask(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return data, mask
    low = np.maximum(coordinates.min(axis=0) - 2, 0)
    high = np.minimum(coordinates.max(axis=0) + 3, np.array(mask.shape))
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(low, high))
    return data[slices], mask[slices]


def orient_panel(array_2d: np.ndarray) -> np.ndarray:
    return np.rot90(array_2d)


def make_mosaic(
    loading_data: np.ndarray,
    brain_mask: np.ndarray,
    component: str,
    output_path: Path,
) -> None:
    loading_data, brain_mask = crop_to_mask(loading_data, brain_mask)
    vmax = float(np.nanmax(np.abs(loading_data)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    planes = [
        ("Sagittal", 0, slice_indices(loading_data.shape[0])),
        ("Coronal", 1, slice_indices(loading_data.shape[1])),
        ("Axial", 2, slice_indices(loading_data.shape[2])),
    ]
    n_columns = max(len(indices) for _, _, indices in planes)
    figure, axes = plt.subplots(3, n_columns, figsize=(15, 9), constrained_layout=True)
    last_image = None

    for row_index, (plane_name, axis, indices) in enumerate(planes):
        for column_index in range(n_columns):
            ax = axes[row_index, column_index]
            ax.axis("off")
            if column_index >= len(indices):
                continue
            index = int(indices[column_index])
            if axis == 0:
                background = brain_mask[index, :, :]
                overlay = loading_data[index, :, :]
            elif axis == 1:
                background = brain_mask[:, index, :]
                overlay = loading_data[:, index, :]
            else:
                background = brain_mask[:, :, index]
                overlay = loading_data[:, :, index]

            background = orient_panel(background.astype(float))
            overlay = orient_panel(overlay)
            ax.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            masked = np.ma.masked_where(np.isclose(overlay, 0.0), overlay)
            last_image = ax.imshow(
                masked,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
                alpha=0.95,
            )
            if column_index == 0:
                ax.set_title(plane_name, fontsize=12, loc="left")

    figure.suptitle(
        f"Whole-brain {component} spatial loading projection",
        fontsize=16,
    )
    if last_image is not None:
        colorbar = figure.colorbar(
            last_image,
            ax=axes,
            orientation="horizontal",
            fraction=0.035,
            pad=0.03,
        )
        colorbar.set_label(f"{component} loading")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_status(status: dict) -> None:
    ensure_dirs([SPATIAL_PROJECTION_OUT, PLOT_SPATIAL_PROJECTION_OUT])
    (SPATIAL_PROJECTION_OUT / "spatial_projection_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


def main() -> None:
    ensure_dirs([SPATIAL_PROJECTION_OUT, PLOT_SPATIAL_PROJECTION_OUT])
    loadings_path = PCA_WHOLE_BRAIN_OUT / "whole_brain_pca_loadings.csv"
    if not loadings_path.exists():
        raise FileNotFoundError(loadings_path)

    loadings = pd.read_csv(loadings_path)
    components = [f"PC{index}" for index in range(1, 6)]
    component_columns = [f"{component}_loading" for component in components]
    required_columns = ["feature", *component_columns]
    missing_columns = [column for column in required_columns if column not in loadings.columns]
    if missing_columns:
        raise KeyError(f"Loadings file is missing columns: {missing_columns}")

    label_mapping = build_label_mapping()
    unmapped_features = sorted(set(loadings["feature"]) - set(label_mapping))
    if unmapped_features:
        raise KeyError(f"No atlas label mapping for features: {unmapped_features}")

    # Save the complete numerical feature-to-atlas mapping even outside the HPC.
    base_manifest_rows = []
    for _, row in loadings.iterrows():
        feature = row["feature"]
        label_id, label_name = label_mapping[feature]
        base_manifest_rows.append(
            {
                "feature": feature,
                "label_id": label_id,
                "label_name": label_name,
                **{column: float(row[column]) for column in component_columns},
            }
        )
    base_manifest = pd.DataFrame(base_manifest_rows)
    base_manifest.to_csv(
        SPATIAL_PROJECTION_OUT / "whole_brain_loading_atlas_mapping_manifest.csv",
        index=False,
    )

    if not SPATIAL_ATLAS_PATH.exists():
        write_status(
            {
                "status": "skipped",
                "reason": "Configured MGZ atlas was not accessible in this runtime",
                "configured_atlas_path": str(SPATIAL_ATLAS_PATH),
                "mapped_features": int(len(base_manifest)),
                "mapping_manifest": str(
                    SPATIAL_PROJECTION_OUT
                    / "whole_brain_loading_atlas_mapping_manifest.csv"
                ),
                "instruction": "Run this script on the HPC or set XTC_SPATIAL_ATLAS to a compatible MGZ path.",
            }
        )
        print(f"Spatial projection skipped: atlas not found at {SPATIAL_ATLAS_PATH}")
        return

    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError(
            "nibabel is required for spatial projection. Run: python -m pip install nibabel"
        ) from error

    atlas_image = nib.load(str(SPATIAL_ATLAS_PATH))
    atlas_data = np.rint(atlas_image.get_fdata(dtype=np.float32)).astype(np.int32)
    unique_labels, voxel_counts = np.unique(atlas_data, return_counts=True)
    count_lookup = dict(zip(unique_labels.tolist(), voxel_counts.tolist()))
    brain_mask_original = atlas_data > 0

    manifest = base_manifest.copy()
    manifest["voxel_count_in_atlas"] = manifest["label_id"].map(
        lambda label_id: int(count_lookup.get(int(label_id), 0))
    )
    manifest["present_in_atlas"] = manifest["voxel_count_in_atlas"] > 0
    manifest.to_csv(
        SPATIAL_PROJECTION_OUT / "whole_brain_loading_atlas_mapping_manifest.csv",
        index=False,
    )

    generated_files = []
    for component, loading_column in zip(components, component_columns):
        loading_volume = np.zeros(atlas_data.shape, dtype=np.float32)
        for _, row in manifest.iterrows():
            loading_volume[atlas_data == int(row["label_id"])] = float(row[loading_column])

        mgz_path = SPATIAL_PROJECTION_OUT / f"whole_brain_{component}_loading_map.mgz"
        nii_path = SPATIAL_PROJECTION_OUT / f"whole_brain_{component}_loading_map.nii.gz"
        nib.save(
            nib.MGHImage(loading_volume, atlas_image.affine, header=atlas_image.header.copy()),
            str(mgz_path),
        )
        nib.save(nib.Nifti1Image(loading_volume, atlas_image.affine), str(nii_path))
        generated_files.extend([str(mgz_path), str(nii_path)])

        loading_image = nib.as_closest_canonical(
            nib.Nifti1Image(loading_volume, atlas_image.affine)
        )
        mask_image = nib.as_closest_canonical(
            nib.Nifti1Image(brain_mask_original.astype(np.float32), atlas_image.affine)
        )
        plot_path = (
            PLOT_SPATIAL_PROJECTION_OUT
            / f"whole_brain_{component}_spatial_loading_projection.png"
        )
        make_mosaic(
            loading_image.get_fdata(dtype=np.float32),
            mask_image.get_fdata(dtype=np.float32) > 0,
            component,
            plot_path,
        )
        generated_files.append(str(plot_path))

    absent = manifest.loc[~manifest["present_in_atlas"], ["feature", "label_id"]]
    write_status(
        {
            "status": "completed",
            "atlas_path": str(SPATIAL_ATLAS_PATH),
            "atlas_shape": list(atlas_data.shape),
            "mapped_features": int(len(manifest)),
            "features_absent_from_this_subject_atlas": absent.to_dict(orient="records"),
            "generated_files": generated_files,
        }
    )
    print(f"Spatial loading projection completed for PC1-PC5 using {SPATIAL_ATLAS_PATH}")


if __name__ == "__main__":
    main()
