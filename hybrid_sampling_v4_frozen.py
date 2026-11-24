
from time import perf_counter

import numpy as np


def _select_spatially_spread(centers, scores, count):
    """Greedy spatial coverage with a modest geometry-score preference."""
    n = len(centers)
    if count >= n:
        return np.arange(n, dtype=np.int64)
    if count <= 0:
        return np.empty(0, dtype=np.int64)

    # Normalize importance to avoid allowing it to dominate spatial coverage.
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - scores.min()
    scores = scores / (scores.max() + 1e-12)

    selected = np.empty(count, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)

    # Start from a geometrically important region.
    first = int(np.argmax(scores))
    selected[0] = first
    used[first] = True

    for i in range(1, count):
        delta = centers - centers[selected[i - 1]]
        d2 = np.sum(delta * delta, axis=1)
        min_dist = np.minimum(min_dist, d2)

        # Spatial spread is primary; importance is a small tie-breaker.
        spread = min_dist / (min_dist.max() + 1e-12)
        priority = spread * (0.85 + 0.15 * scores)
        priority[used] = -np.inf

        nxt = int(np.argmax(priority))
        selected[i] = nxt
        used[nxt] = True

    return selected


def hybrid_sample_v4(
    xyz,
    target_points=9416,
    voxel_size=0.8,
    random_state=0,
):
    """
    Geometry-aware spatial voxel sampler.

    - Uses only XYZ geometry; labels are never read.
    - Returns actual original points, not synthetic voxel centroids.
    - Tries to maintain spatial coverage while favoring complex voxels.
    - Returns exactly target_points when enough input points exist.
    """
    start = perf_counter()

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("xyz must have shape (N, 3) or (N, >=3)")

    xyz = xyz[:, :3]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]

    n = len(xyz)
    target = min(max(int(target_points), 0), n)

    if target == 0:
        return np.empty((0, 3), dtype=np.float32)
    if target == n:
        return xyz.copy()
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    # Build voxel groups.
    voxel_keys = np.floor(xyz / voxel_size).astype(np.int32)
    unique_keys, inverse = np.unique(
        voxel_keys, axis=0, return_inverse=True
    )
    voxel_count = len(unique_keys)

    order = np.argsort(inverse, kind="stable")
    boundaries = np.r_[
        0,
        np.flatnonzero(np.diff(inverse[order])) + 1,
        n,
    ]

    centers = np.empty((voxel_count, 3), dtype=np.float32)
    representatives = np.empty(voxel_count, dtype=np.int64)
    secondary = np.full(voxel_count, -1, dtype=np.int64)
    complexity = np.zeros(voxel_count, dtype=np.float32)
    counts = np.zeros(voxel_count, dtype=np.int32)

    # One centroid-nearest point and one geometrically distinct point
    # per voxel where possible.
    for voxel_id, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        ids = order[left:right]
        points = xyz[ids]
        center = points.mean(axis=0)

        centers[voxel_id] = center
        counts[voxel_id] = len(ids)

        d2 = np.sum((points - center) ** 2, axis=1)
        representatives[voxel_id] = ids[int(np.argmin(d2))]

        if len(ids) > 1:
            secondary[voxel_id] = ids[int(np.argmax(d2))]

        # Geometry spread plus a small density contribution.
        spread = float(np.mean(d2))
        complexity[voxel_id] = (
            np.sqrt(max(spread, 0.0))
            + 0.02 * np.log1p(len(ids))
        )

    rng = np.random.default_rng(random_state)

    if target <= voxel_count:
        # Preserve spatial distribution when the budget is smaller
        # than the number of occupied voxels.
        chosen_voxels = _select_spatially_spread(
            centers, complexity, target
        )
        selected_indices = representatives[chosen_voxels]

    else:
        # First reserve one representative from every occupied voxel.
        selected_indices = list(representatives)

        remaining = target - voxel_count

        # Prefer second points from geometrically varied voxels.
        valid_voxels = np.flatnonzero(secondary >= 0)

        if remaining > 0 and len(valid_voxels):
            secondary_scores = complexity[valid_voxels]
            ranked = valid_voxels[
                np.argsort(secondary_scores)[::-1]
            ]

            take = min(remaining, len(ranked))
            selected_indices.extend(secondary[ranked[:take]])
            remaining -= take

        # If the target is still not reached, fill with remaining
        # original points, preferring voxels with higher spread.
        if remaining > 0:
            already = np.zeros(n, dtype=bool)
            already[np.asarray(selected_indices, dtype=np.int64)] = True
            available = np.flatnonzero(~already)

            if len(available):
                point_voxels = inverse[available]
                point_scores = complexity[point_voxels]

                # Small seeded jitter breaks ties reproducibly.
                point_scores = (
                    point_scores
                    + rng.random(len(available)) * 1e-7
                )
                ranked = available[np.argsort(point_scores)[::-1]]
                take = min(remaining, len(ranked))
                selected_indices.extend(ranked[:take])

        selected_indices = np.asarray(
            selected_indices, dtype=np.int64
        )

        # Defensive exact-budget correction.
        selected_indices = np.unique(selected_indices)
        if len(selected_indices) > target:
            selected_indices = selected_indices[:target]
        elif len(selected_indices) < target:
            already = np.zeros(n, dtype=bool)
            already[selected_indices] = True
            available = np.flatnonzero(~already)
            take = min(target - len(selected_indices), len(available))
            selected_indices = np.concatenate(
                (selected_indices, available[:take])
            )

    result = xyz[selected_indices]

    print(f"Original points: {n}")
    print(f"Occupied voxels: {voxel_count}")
    print(f"Target points: {target}")
    print(f"Output points: {len(result)}")
    print(f"Total wall time: {perf_counter() - start:.3f} s")

    return result.astype(np.float32)
