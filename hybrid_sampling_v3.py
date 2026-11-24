
from time import perf_counter

import numpy as np

from adaptive_sampling_v3 import adaptive_sample_v3


def _farthest_point_sampling(points, count):
    """Select actual input points, spreading them across the candidate cloud."""
    n = len(points)

    if count >= n:
        return np.arange(n, dtype=np.int64)

    selected = np.empty(count, dtype=np.int64)
    min_dist = np.full(n, np.inf, dtype=np.float32)

    # Start with the point farthest from the cloud centroid.
    centroid = points.mean(axis=0)
    selected[0] = np.argmax(
        np.sum((points - centroid) ** 2, axis=1)
    )

    for i in range(count):
        current = selected[i]
        dist = np.sum((points - points[current]) ** 2, axis=1)
        np.minimum(min_dist, dist, out=min_dist)

        if i + 1 < count:
            min_dist[selected[:i + 1]] = -1
            selected[i + 1] = np.argmax(min_dist)

    return selected


def hybrid_sample_v3(
    xyz,
    target_points=9416,
    voxel_size=0.8,
    random_state=0,
):
    """
    Experimental hybrid:
    - V3 geometric sampling
    - voxel-based boundary candidate enrichment
    - final selection from original LiDAR points only

    KITTI labels are never used.
    """
    start = perf_counter()

    xyz = np.asarray(xyz, dtype=np.float32)
    xyz = xyz[np.all(np.isfinite(xyz[:, :3]), axis=1), :3]

    n = len(xyz)
    target = min(max(int(target_points), 1), n)

    if target == n:
        return xyz.copy()

    rng = np.random.default_rng(random_state)

    # Stage 1: V3 geometric candidates.
    v3_points = adaptive_sample_v3(
        xyz, target_points=target
    )
    v3_points = np.asarray(v3_points, dtype=np.float32)[:, :3]

    # Stage 2: Generate actual-point candidates near voxel boundaries.
    voxel_keys = np.floor(xyz / voxel_size).astype(np.int32)
    _, inverse = np.unique(
        voxel_keys, axis=0, return_inverse=True
    )

    order = np.argsort(inverse, kind="stable")
    boundaries = np.r_[
        0,
        np.flatnonzero(np.diff(inverse[order])) + 1,
        n,
    ]

    extra_indices = []

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        ids = order[left:right]

        # Keep the centroid-nearest actual point in each voxel.
        local = xyz[ids]
        center = local.mean(axis=0)
        nearest = ids[np.argmin(
            np.sum((local - center) ** 2, axis=1)
        )]
        extra_indices.append(nearest)

        # In larger voxels, add a point far from the center.
        if len(ids) >= 8:
            d2 = np.sum((local - center) ** 2, axis=1)
            extra_indices.append(ids[np.argmax(d2)])

    extra_indices = np.asarray(extra_indices, dtype=np.int64)
    extra_points = xyz[extra_indices]

    # Stage 3: Union V3 and voxel candidates.
    candidates = np.vstack((v3_points, extra_points))

    # Match candidates to actual input points so output contains no
    # synthetic centroids. Round only for deduplication.
    _, unique_idx = np.unique(
        np.round(candidates, decimals=5),
        axis=0,
        return_index=True,
    )
    candidates = candidates[np.sort(unique_idx)]

    # Match each candidate to its nearest original point.
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    _, original_idx = tree.query(candidates, k=1)
    original_idx = np.unique(original_idx)

    # If needed, add original points as fallback candidates.
    if len(original_idx) < target:
        remaining = np.setdiff1d(
            np.arange(n), original_idx, assume_unique=True
        )
        extra_count = min(target - len(original_idx), len(remaining))
        if extra_count:
            fallback = rng.choice(
                remaining, size=extra_count, replace=False
            )
            original_idx = np.concatenate((original_idx, fallback))

    pool = xyz[original_idx]

    # Stage 4: Spread the final selection across the candidate pool.
    if len(pool) > target:
        selected = _farthest_point_sampling(pool, target)
        result = pool[selected]
    else:
        result = pool

    elapsed = perf_counter() - start

    print(f"Original points: {n}")
    print(f"Candidate points: {len(pool)}")
    print(f"Target points: {target}")
    print(f"Output points: {len(result)}")
    print(f"Total wall time: {elapsed:.3f} s")

    return result.astype(np.float32)
