
import numpy as np

from adaptive_sampling_v3 import adaptive_sample_v3
from adaptive_sampling_self_adaptive_grid import adaptive_sample_self_adaptive_grid


def _farthest_point_sampling(points, count):
    """Select spatially distributed points from a candidate set."""
    n = len(points)

    if count >= n:
        return points.copy()

    selected = np.empty(count, dtype=np.int64)
    selected[0] = 0

    min_dist = np.sum((points - points[0]) ** 2, axis=1)

    for i in range(1, count):
        idx = int(np.argmax(min_dist))
        selected[i] = idx

        dist = np.sum((points - points[idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected[:i + 1]] = -1

    return points[selected]


def hybrid_sample(
    xyz,
    target_points=9416,
    voxel_size=0.8,
    random_state=0,
    ransac_iterations=5,
    max_split_depth=2,
    normal_k=12,
):
    """
    Hybrid sampler combining V3 and self-adaptive grid sampling.

    The reduced RANSAC iterations, split depth and normal neighborhood
    are experimental speed settings. Their effect on quality must be tested.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    xyz = xyz[np.all(np.isfinite(xyz[:, :3]), axis=1), :3]

    n = len(xyz)
    target = min(max(int(target_points), 1), n)

    if target == n:
        return xyz.copy()

    # Method 1: V3 adaptive sampler
    v3_points = adaptive_sample_v3(
        xyz,
        target_points=target,
    )

    # Method 2: self-adaptive grid sampler
    grid_points = adaptive_sample_self_adaptive_grid(
        xyz,
        target_points=target,
        voxel_size=voxel_size,
        random_state=random_state,
        ransac_iterations=ransac_iterations,
        max_split_depth=max_split_depth,
        normal_k=normal_k,
    )

    # Merge both methods' outputs.
    candidates = np.vstack((v3_points, grid_points))
    candidates = np.unique(np.round(candidates, decimals=5), axis=0)

    # Enforce the requested point budget.
    if len(candidates) > target:
        result = _farthest_point_sampling(candidates, target)
    else:
        result = candidates

    return result.astype(np.float32)
