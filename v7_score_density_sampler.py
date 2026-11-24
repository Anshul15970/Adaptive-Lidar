import argparse
import csv
from pathlib import Path

import numpy as np

from v6_density_sampler import allocate_quotas


VOXEL_SIZE = 0.8
LAMBDAS = [0.25, 0.50, 0.75]

PROJECT_ROOT = Path("/mnt/e/Projects/adaptive-lidar")

INPUT_DIR = (
    PROJECT_ROOT
    / "data/object_detection/training/velodyne"
)

VAL_FILE = (
    Path.home()
    / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

OUTPUT_ROOT = PROJECT_ROOT / "data/detector_sampling"


def calculate_v5_scores(xyz, inverse, counts):
    """
    Reproduce V5's score calculation exactly:
    normalized inverse-sqrt occupancy and normalized complexity.
    """
    voxel_count = len(counts)

    sum_x = np.bincount(
        inverse,
        weights=xyz[:, 0],
        minlength=voxel_count
    )

    sum_y = np.bincount(
        inverse,
        weights=xyz[:, 1],
        minlength=voxel_count
    )

    sum_z = np.bincount(
        inverse,
        weights=xyz[:, 2],
        minlength=voxel_count
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

    complexity = np.sqrt(
        np.maximum(mean_distances_sq, 0.0)
    )

    sparsity = 1.0 / np.sqrt(
        counts.astype(np.float64)
    )

    sparsity = (
        (sparsity - sparsity.min())
        / (sparsity.max() - sparsity.min() + 1e-12)
    )

    complexity = (
        (complexity - complexity.min())
        / (complexity.max() - complexity.min() + 1e-12)
    )

    score = 0.65 * sparsity + 0.35 * complexity

    return centers, score


def sample_v7_indices(
    xyz,
    target_points,
    density_weight,
    voxel_size=VOXEL_SIZE
):
    """
    V7 score-density weighted quota sampler.

    Returns indices into the original supplied XYZ array.
    No ground-truth labels are used.
    """

    xyz = np.asarray(xyz, dtype=np.float32)

    if not 0.0 < density_weight < 1.0:
        raise ValueError("density_weight must be between 0 and 1.")

    finite_mask = np.isfinite(xyz).all(axis=1)

    valid_indices = np.flatnonzero(finite_mask)
    valid_xyz = xyz[finite_mask]

    if target_points > len(valid_xyz):
        raise ValueError("Target exceeds valid point count.")

    if target_points == 0:
        return np.empty(0, dtype=np.int64)

    voxel_keys = np.floor(
        valid_xyz / voxel_size
    ).astype(np.int64)

    _, inverse = np.unique(
        voxel_keys,
        axis=0,
        return_inverse=True
    )

    voxel_count = int(inverse.max()) + 1

    if target_points < voxel_count:
        raise ValueError(
            f"Target ({target_points}) is smaller than "
            f"occupied voxel count ({voxel_count})."
        )

    order = np.argsort(inverse, kind="stable")

    counts = np.bincount(
        inverse,
        minlength=voxel_count
    )

    starts = np.concatenate([
        [0],
        np.cumsum(counts)[:-1]
    ])

    centers, scores = calculate_v5_scores(
        valid_xyz,
        inverse,
        counts
    )

    # Normalize both allocation signals to mean 1.
    score_weights = scores / (
        scores.mean() + 1e-12
    )

    density_weights = counts.astype(np.float64)

    density_weights /= (
        density_weights.mean() + 1e-12
    )

    weights = (
        (1.0 - density_weight) * score_weights
        + density_weight * density_weights
    )

    representatives = np.empty(
        voxel_count,
        dtype=np.int64
    )

    capacities = counts - 1

    additional_orders = []

    for voxel_id in range(voxel_count):

        start = starts[voxel_id]
        end = start + counts[voxel_id]

        local_indices = order[start:end]
        local_points = valid_xyz[local_indices]

        distances_sq = np.sum(
            (local_points - centers[voxel_id]) ** 2,
            axis=1
        )

        # One representative nearest to the centroid.
        nearest_position = np.argmin(distances_sq)

        representatives[voxel_id] = (
            local_indices[nearest_position]
        )

        # Remaining points, farthest from centroid first.
        remaining_mask = np.ones(
            len(local_indices),
            dtype=bool
        )

        remaining_mask[nearest_position] = False

        remaining_indices = local_indices[remaining_mask]
        remaining_distances = distances_sq[remaining_mask]

        local_order = np.lexsort(
            (remaining_indices, -remaining_distances)
        )

        additional_orders.append(
            remaining_indices[local_order]
        )

    extra_budget = target_points - voxel_count

    quotas = allocate_quotas(
        capacities=capacities,
        weights=weights,
        total=extra_budget
    )

    selected = list(representatives)

    for voxel_id, quota in enumerate(quotas):

        if quota > 0:
            selected.extend(
                additional_orders[voxel_id][:quota].tolist()
            )

    selected = np.asarray(
        selected,
        dtype=np.int64
    )

    if len(selected) != target_points:
        raise RuntimeError(
            f"Budget mismatch: expected {target_points}, "
            f"got {len(selected)}"
        )

    if len(np.unique(selected)) != target_points:
        raise RuntimeError("Duplicate point indices detected.")

    return valid_indices[selected]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames",
        type=int,
        default=50
    )

    args = parser.parse_args()

    frame_ids = VAL_FILE.read_text().split()
    frame_ids = frame_ids[:args.frames]

    summary_path = (
        OUTPUT_ROOT / "v7_score_density_summary.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for frame_number, frame_id in enumerate(
        frame_ids,
        start=1
    ):

        input_path = INPUT_DIR / f"{frame_id}.bin"

        if not input_path.exists():
            print(f"[SKIP] Missing frame: {frame_id}")
            continue

        raw = np.fromfile(
            input_path,
            dtype=np.float32
        )

        if raw.size % 4 != 0:
            raise ValueError(
                f"Invalid XYZI file: {input_path}"
            )

        raw = raw.reshape(-1, 4)

        xyz = raw[:, :3]

        # Match V5 and V6 budget calculation.
        target = len(raw) // 5

        print(
            f"\n[{frame_number}/{len(frame_ids)}] "
            f"Frame {frame_id}: {len(raw)} points, "
            f"target={target}"
        )

        for lam in LAMBDAS:

            selected_indices = sample_v7_indices(
                xyz=xyz,
                target_points=target,
                density_weight=lam,
                voxel_size=VOXEL_SIZE
            )

            # Original XYZI rows are selected by index.
            sampled_xyzi = raw[selected_indices]

            variant = f"v7_lambda_{lam:.2f}"

            output_dir = (
                OUTPUT_ROOT / variant / "velodyne"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path = (
                output_dir / f"{frame_id}.bin"
            )

            sampled_xyzi.astype(
                np.float32
            ).tofile(output_path)

            rows.append({
                "frame_id": frame_id,
                "lambda": lam,
                "original_points": len(raw),
                "target_points": target,
                "sampled_points": len(sampled_xyzi),
                "retention_percent": (
                    100.0 * len(sampled_xyzi) / len(raw)
                ),
            })

            print(
                f"  lambda={lam:.2f}: "
                f"{len(sampled_xyzi)} points exported"
            )

    with open(
        summary_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "lambda",
                "original_points",
                "target_points",
                "sampled_points",
                "retention_percent",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nV7 sampling complete.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
