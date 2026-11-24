
import numpy as np
from features import compute_features


def adaptive_sample_v2(xyz, target_points=9416, k=20,
                       complexity_weight=1.0):
    """
    Geometry-only adaptive voxel sampling.
    Uses no labels or class information.
    Returns exactly target_points when feasible.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    n = len(xyz)

    if target_points >= n:
        return xyz.copy()

    if target_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    # 1. Estimate geometric complexity
    scores = compute_features(xyz, k=k)

    # Normalize scores to [0, 1]
    max_score = scores.max()
    if max_score > 1e-12:
        scores = scores / max_score

    # 2. Find a voxel size yielding at least the target
    # number of occupied cells.
    lower = 1e-3
    upper = max(float(np.ptp(xyz, axis=0).max()), 1.0)

    for _ in range(18):
        voxel_size = (lower + upper) / 2
        cells = np.floor(xyz / voxel_size).astype(np.int32)
        _, inverse = np.unique(
            cells, axis=0, return_inverse=True
        )
        cell_count = inverse.max() + 1

        if cell_count >= target_points:
            lower = voxel_size
        else:
            upper = voxel_size

    # Use the finer boundary to ensure enough candidates.
    voxel_size = lower
    cells = np.floor(xyz / voxel_size).astype(np.int32)
    _, inverse = np.unique(
        cells, axis=0, return_inverse=True
    )
    cell_count = inverse.max() + 1

    # 3. Choose one representative per voxel.
    # Prefer central points, with a modest complexity bonus.
    counts = np.bincount(inverse, minlength=cell_count)

    centers = np.column_stack([
        np.bincount(inverse, weights=xyz[:, dim],
                    minlength=cell_count) / counts
        for dim in range(3)
    ])

    squared_distance = np.sum(
        (xyz - centers[inverse]) ** 2, axis=1
    )

    # Centrality keeps spatial coverage; complexity favours detail.
    priority = (
        squared_distance /
        (1.0 + complexity_weight * scores)
    )

    # Select the best representative within each voxel.
    order = np.lexsort((priority, inverse))
    sorted_cells = inverse[order]
    first = np.r_[True, sorted_cells[1:] != sorted_cells[:-1]]
    representatives = order[first]

    # 4. Keep exactly the target count if there are extra cells.
    if len(representatives) > target_points:
        rep_scores = scores[representatives]
        rep_dist = squared_distance[representatives]
        rep_priority = (
            rep_dist / (1.0 + complexity_weight * rep_scores)
        )
        keep = np.argsort(rep_priority)[:target_points]
        representatives = representatives[keep]

    # If numerical edge cases leave too few points, fill from
    # unselected points with the highest complexity scores.
    if len(representatives) < target_points:
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[representatives] = True
        remaining = np.flatnonzero(~selected_mask)
        needed = target_points - len(representatives)
        extra = remaining[
            np.argsort(scores[remaining])[-needed:]
        ]
        representatives = np.concatenate([representatives, extra])

    return xyz[representatives]
