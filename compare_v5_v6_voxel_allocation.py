
from pathlib import Path
import csv

import numpy as np
from tqdm import tqdm

from class_retention_detector_clouds import (
    load_xyz,
    read_calibration,
    lidar_to_camera,
    read_boxes,
    points_in_box,
)


ROOT = Path("/mnt/e/Projects/adaptive-lidar")

ORIGINAL = ROOT / "data/object_detection/training/velodyne"
LABELS = ROOT / "data/object_detection/training/label_2"
CALIB = ROOT / "data/object_detection/training/calib"

METHODS = {
    "v5": ROOT / "data/detector_sampling/hybrid_v5/velodyne",
    "v6_a00": ROOT / "data/detector_sampling/v6_alpha_0.00/velodyne",
    "v6_a025": ROOT / "data/detector_sampling/v6_alpha_0.25/velodyne",
    "v6_a050": ROOT / "data/detector_sampling/v6_alpha_0.50/velodyne",
    "v6_a075": ROOT / "data/detector_sampling/v6_alpha_0.75/velodyne",
}

SPLIT = (
    Path.home()
    / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

VOXEL_SIZE = 0.8

DENSITY_NAMES = [
    "1-5",
    "6-10",
    "11-20",
    "21-50",
    "51+",
]

CLASSES = ["Car", "Pedestrian", "Cyclist"]

OUTPUT = ROOT / "results/v5_v6_voxel_allocation_50.csv"


def get_density_bins(counts):
    """Assign each voxel to an occupancy category."""
    return np.select(
        [
            counts <= 5,
            counts <= 10,
            counts <= 20,
            counts <= 50,
        ],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int64)


def calculate_v5_scores(xyz, inverse, counts):
    """
    Reproduce V5's sparsity, complexity and combined score.
    """
    voxel_count = len(counts)

    sum_x = np.bincount(
        inverse, weights=xyz[:, 0], minlength=voxel_count
    )
    sum_y = np.bincount(
        inverse, weights=xyz[:, 1], minlength=voxel_count
    )
    sum_z = np.bincount(
        inverse, weights=xyz[:, 2], minlength=voxel_count
    )

    centers = np.column_stack([
        sum_x / counts,
        sum_y / counts,
        sum_z / counts,
    ])

    distances_sq = np.sum(
        (xyz - centers[inverse]) ** 2,
        axis=1
    )

    mean_distances_sq = (
        np.bincount(
            inverse,
            weights=distances_sq,
            minlength=voxel_count
        ) / counts
    )

    complexity = np.sqrt(np.maximum(mean_distances_sq, 0.0))

    sparsity = 1.0 / np.sqrt(counts.astype(np.float64))

    sparsity = (
        (sparsity - sparsity.min())
        / (sparsity.max() - sparsity.min() + 1e-12)
    )

    complexity = (
        (complexity - complexity.min())
        / (complexity.max() - complexity.min() + 1e-12)
    )

    score = 0.65 * sparsity + 0.35 * complexity

    return sparsity, complexity, score


def get_class_masks(points_camera, boxes):
    """Create independent point masks for each evaluated class."""
    masks = {
        cls: np.zeros(len(points_camera), dtype=bool)
        for cls in CLASSES
    }

    for box in boxes:
        cls = box["class"]

        if cls not in masks:
            continue

        masks[cls] |= points_in_box(points_camera, box)

    return masks


def aggregate(values, group_ids, group_count):
    return np.bincount(
        group_ids,
        weights=np.asarray(values, dtype=np.float64),
        minlength=group_count
    )


def main():
    frame_ids = SPLIT.read_text().split()[:50]

    rows = []

    density_count = len(DENSITY_NAMES)
    score_count = 5
    group_count = density_count * score_count

    for frame_id in tqdm(frame_ids, desc="Voxel allocation analysis"):

        original_path = ORIGINAL / f"{frame_id}.bin"

        raw = np.fromfile(original_path, dtype=np.float32)
        raw = raw.reshape(-1, 4)

        original_xyz = raw[:, :3]
        finite_mask = np.isfinite(original_xyz).all(axis=1)

        original_xyz = original_xyz[finite_mask]

        rotation, translation = read_calibration(
            CALIB / f"{frame_id}.txt"
        )

        original_camera = lidar_to_camera(
            original_xyz, rotation, translation
        )

        boxes = read_boxes(LABELS / f"{frame_id}.txt")

        original_masks = get_class_masks(
            original_camera, boxes
        )

        # Background means outside all evaluated object boxes.
        object_union = np.zeros(len(original_xyz), dtype=bool)

        for mask in original_masks.values():
            object_union |= mask

        original_masks["Background"] = ~object_union

        # Load every sampled cloud.
        sampled_xyz = {}

        for method, folder in METHODS.items():
            path = folder / f"{frame_id}.bin"

            if not path.exists():
                raise FileNotFoundError(path)

            sampled_xyz[method] = load_xyz(path)

        # Build common voxel IDs across original and sampled clouds.
        original_keys = np.floor(
            original_xyz / VOXEL_SIZE
        ).astype(np.int32)

        key_arrays = [original_keys]

        for method in METHODS:
            keys = np.floor(
                sampled_xyz[method] / VOXEL_SIZE
            ).astype(np.int32)

            key_arrays.append(keys)

        lengths = [len(x) for x in key_arrays]

        all_keys = np.vstack(key_arrays)

        _, all_inverse = np.unique(
            all_keys,
            axis=0,
            return_inverse=True
        )

        offsets = np.cumsum([0] + lengths)

        original_inverse = all_inverse[
            offsets[0]:offsets[1]
        ]

        voxel_count = int(original_inverse.max()) + 1

        counts = np.bincount(
            original_inverse,
            minlength=voxel_count
        )

        sparsity, complexity, score = calculate_v5_scores(
            original_xyz,
            original_inverse,
            counts
        )

        # Group voxels by occupancy and V5 score quintile.
        density_bins = get_density_bins(counts)

        score_edges = np.quantile(
            score, [0.2, 0.4, 0.6, 0.8]
        )

        score_bins = np.searchsorted(
            score_edges, score, side="right"
        )

        group_ids = density_bins * score_count + score_bins

        # Per-group original voxel and point counts.
        voxel_totals = aggregate(
            np.ones(voxel_count),
            group_ids,
            group_count
        )

        original_point_totals = aggregate(
            counts,
            group_ids,
            group_count
        )

        # Original object/background points per group.
        original_category_totals = {}

        for category, mask in original_masks.items():
            per_voxel = np.bincount(
                original_inverse[mask],
                minlength=voxel_count
            )

            original_category_totals[category] = aggregate(
                per_voxel,
                group_ids,
                group_count
            )

        # Sampled point counts per group and category.
        sampled_totals = {}
        sampled_category_totals = {}

        for method_index, method in enumerate(METHODS):

            start = offsets[method_index + 1]
            end = offsets[method_index + 2]

            sampled_inverse = all_inverse[start:end]

            sampled_totals[method] = aggregate(
                np.bincount(
                    sampled_inverse,
                    minlength=voxel_count
                ),
                group_ids,
                group_count
            )

            sampled_camera = lidar_to_camera(
                sampled_xyz[method],
                rotation,
                translation
            )

            sampled_masks = get_class_masks(
                sampled_camera, boxes
            )

            sampled_union = np.zeros(
                len(sampled_camera), dtype=bool
            )

            for mask in sampled_masks.values():
                sampled_union |= mask

            sampled_masks["Background"] = ~sampled_union

            sampled_category_totals[method] = {}

            for category, mask in sampled_masks.items():

                per_voxel = np.bincount(
                    sampled_inverse[mask],
                    minlength=voxel_count
                )

                sampled_category_totals[method][category] = (
                    aggregate(
                        per_voxel,
                        group_ids,
                        group_count
                    )
                )

        # Save aggregated results.
        for density_index, density_name in enumerate(
            DENSITY_NAMES
        ):
            for score_index in range(score_count):

                group = density_index * score_count + score_index

                if voxel_totals[group] == 0:
                    continue

                row = {
                    "frame_id": frame_id,
                    "density_bin": density_name,
                    "v5_score_quintile": score_index + 1,
                    "voxel_count": int(voxel_totals[group]),
                    "original_points": int(
                        original_point_totals[group]
                    ),
                }

                for method in METHODS:
                    selected = sampled_totals[method][group]

                    row[f"{method}_selected"] = int(selected)

                    row[f"{method}_retention_percent"] = (
                        100.0 * selected
                        / original_point_totals[group]
                        if original_point_totals[group] else 0.0
                    )

                    for category in (
                        "Car",
                        "Pedestrian",
                        "Cyclist",
                        "Background",
                    ):
                        original_category = (
                            original_category_totals[category][group]
                        )

                        selected_category = (
                            sampled_category_totals[method][category][group]
                        )

                        row[
                            f"{method}_{category.lower()}_original"
                        ] = int(original_category)

                        row[
                            f"{method}_{category.lower()}_selected"
                        ] = int(selected_category)

                        row[
                            f"{method}_{category.lower()}_retention"
                        ] = (
                            100.0 * selected_category
                            / original_category
                            if original_category else 0.0
                        )

                rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with open(OUTPUT, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\n========== VOXEL ALLOCATION DIAGNOSTICS ==========")
    print(f"Frames processed: {len(frame_ids)}")
    print(f"Aggregated rows: {len(rows)}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

