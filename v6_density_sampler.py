import argparse
import csv
from pathlib import Path

import numpy as np


VOXEL_SIZE = 0.8
ALPHAS = [0.00, 0.25, 0.50, 0.75]

PROJECT_ROOT = Path("/mnt/e/Projects/adaptive-lidar")
INPUT_DIR = PROJECT_ROOT / "data/object_detection/training/velodyne"
VAL_FILE = (
    Path.home()
    / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

OUTPUT_ROOT = PROJECT_ROOT / "data/detector_sampling"


def allocate_quotas(capacities, weights, total):
    """
    Allocate exactly `total` additional points across voxels.

    Uses weighted proportional allocation with capacity limits
    and deterministic largest-remainder handling.
    """
    capacities = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    quotas = np.zeros(len(capacities), dtype=np.int64)
    remaining = int(total)

    if remaining < 0 or remaining > capacities.sum():
        raise ValueError("Requested allocation exceeds available capacity.")

    while remaining > 0:
        active = np.flatnonzero(quotas < capacities)

        if len(active) == 0:
            raise RuntimeError("No voxel capacity left.")

        active_weights = weights[active]
        ideal = remaining * active_weights / active_weights.sum()

        proposed = np.floor(ideal).astype(np.int64)
        available = capacities[active] - quotas[active]
        proposed = np.minimum(proposed, available)

        assigned = int(proposed.sum())

        if assigned > 0:
            quotas[active] += proposed
            remaining -= assigned
            continue

        # Every ideal quota is below one.
        # Assign the remaining points by largest fractional remainder.
        fractions = ideal - np.floor(ideal)
        order = np.lexsort((active, -fractions))

        for position in order:
            voxel = active[position]

            if remaining == 0:
                break

            if quotas[voxel] < capacities[voxel]:
                quotas[voxel] += 1
                remaining -= 1

    return quotas


def sample_v6_indices(xyz, target_points, alpha, voxel_size=VOXEL_SIZE):
    """
    Returns indices into the supplied XYZ array.

    The sampler uses no ground-truth labels.
    """
    xyz = np.asarray(xyz, dtype=np.float32)

    finite_mask = np.isfinite(xyz).all(axis=1)
    valid_indices = np.flatnonzero(finite_mask)
    valid_xyz = xyz[finite_mask]

    if target_points > len(valid_xyz):
        raise ValueError("Target exceeds the number of valid points.")

    if target_points == 0:
        return np.empty(0, dtype=np.int64)

    voxel_keys = np.floor(valid_xyz / voxel_size).astype(np.int64)

    unique_keys, inverse = np.unique(
        voxel_keys,
        axis=0,
        return_inverse=True
    )

    voxel_count = len(unique_keys)

    # This first experiment assumes at least one point can be
    # selected from every occupied voxel.
    if target_points < voxel_count:
        raise ValueError(
            f"Target ({target_points}) is smaller than occupied "
            f"voxel count ({voxel_count})."
        )

    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=voxel_count)

    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    representatives = np.empty(voxel_count, dtype=np.int64)
    capacities = counts - 1

    # Store each voxel's additional-point ordering.
    additional_orders = []

    for voxel_id in range(voxel_count):
        start = starts[voxel_id]
        end = start + counts[voxel_id]

        local_indices = order[start:end]
        local_points = valid_xyz[local_indices]

        centroid = local_points.mean(axis=0)
        distances_sq = np.sum(
            (local_points - centroid) ** 2,
            axis=1
        )

        # Nearest-to-centroid point is the representative.
        nearest_position = np.argmin(distances_sq)
        representative = local_indices[nearest_position]

        representatives[voxel_id] = representative

        # Additional points: farthest from centroid first.
        remaining_mask = np.ones(len(local_indices), dtype=bool)
        remaining_mask[nearest_position] = False

        remaining_indices = local_indices[remaining_mask]
        remaining_distances = distances_sq[remaining_mask]

        local_order = np.lexsort(
            (remaining_indices, -remaining_distances)
        )

        additional_orders.append(remaining_indices[local_order])

    extra_budget = target_points - voxel_count

    weights = np.power(counts.astype(np.float64), alpha)

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

    selected = np.asarray(selected, dtype=np.int64)

    if len(selected) != target_points:
        raise RuntimeError(
            f"Budget mismatch: expected {target_points}, "
            f"got {len(selected)}"
        )

    if len(np.unique(selected)) != target_points:
        raise RuntimeError("Duplicate point indices detected.")

    # Convert valid-array indices back to original XYZ-array indices.
    return valid_indices[selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()

    with open(VAL_FILE, "r") as file:
        frame_ids = [line.strip() for line in file if line.strip()]

    frame_ids = frame_ids[:args.frames]

    summary_path = OUTPUT_ROOT / "v6_density_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for frame_number, frame_id in enumerate(frame_ids, start=1):
        input_path = INPUT_DIR / f"{frame_id}.bin"

        if not input_path.exists():
            print(f"[SKIP] Missing frame: {frame_id}")
            continue

        raw = np.fromfile(input_path, dtype=np.float32)

        if raw.size % 4 != 0:
            raise ValueError(f"Invalid XYZI file: {input_path}")

        raw = raw.reshape(-1, 4)
        xyz = raw[:, :3]

        # Match V5's target calculation.
        target = len(raw) // 5

        print(
            f"\n[{frame_number}/{len(frame_ids)}] "
            f"Frame {frame_id}: {len(raw)} points, "
            f"target={target}"
        )

        for alpha in ALPHAS:
            selected_indices = sample_v6_indices(
                xyz=xyz,
                target_points=target,
                alpha=alpha,
                voxel_size=VOXEL_SIZE
            )

            sampled_xyzi = raw[selected_indices]

            variant = f"v6_alpha_{alpha:.2f}"
            output_dir = OUTPUT_ROOT / variant / "velodyne"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{frame_id}.bin"
            sampled_xyzi.astype(np.float32).tofile(output_path)

            rows.append({
                "frame_id": frame_id,
                "alpha": alpha,
                "original_points": len(raw),
                "target_points": target,
                "sampled_points": len(sampled_xyzi),
                "retention_percent": (
                    100.0 * len(sampled_xyzi) / len(raw)
                ),
            })

            print(
                f"  alpha={alpha:.2f}: "
                f"{len(sampled_xyzi)} points exported"
            )

    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "alpha",
                "original_points",
                "target_points",
                "sampled_points",
                "retention_percent",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nV6 sampling complete.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
