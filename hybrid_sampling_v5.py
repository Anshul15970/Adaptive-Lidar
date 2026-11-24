from time import perf_counter
import numpy as np


def _select_spatially_spread(centers, scores, count):
    n = len(centers)
    if count >= n:
        return np.arange(n, dtype=np.int64)
    if count <= 0:
        return np.empty(0, dtype=np.int64)

    centers = np.asarray(centers, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    scores = scores - scores.min()
    scores = scores / (scores.max() + 1e-12)

    selected = np.empty(count, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)

    first = int(np.argmax(scores))
    selected[0] = first
    used[first] = True

    for i in range(1, count):
        delta = centers - centers[selected[i - 1]]
        d2 = np.einsum("ij,ij->i", delta, delta)
        np.minimum(min_dist, d2, out=min_dist)

        spread = min_dist / (min_dist.max() + 1e-12)
        priority = spread * (0.65 + 0.35 * scores)
        priority[used] = -np.inf

        nxt = int(np.argmax(priority))
        selected[i] = nxt
        used[nxt] = True

    return selected


def hybrid_sample_v5(
    xyz,
    target_points=9416,
    voxel_size=0.8,
    random_state=0,
    verbose=False,
):
    """Label-free voxel sampler with extra priority for sparse regions."""
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

    voxel_keys = np.floor(xyz / voxel_size).astype(np.int32)
    unique_keys, inverse, counts = np.unique(
        voxel_keys, axis=0, return_inverse=True, return_counts=True
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
    voxel_sizes = np.zeros(voxel_count, dtype=np.int32)

    for voxel_id, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        ids = order[left:right]
        points = xyz[ids]
        center = points.mean(axis=0)

        centers[voxel_id] = center
        voxel_sizes[voxel_id] = len(ids)

        d2 = np.sum((points - center) ** 2, axis=1)
        representatives[voxel_id] = ids[int(np.argmin(d2))]

        if len(ids) > 1:
            secondary[voxel_id] = ids[int(np.argmax(d2))]

        spread = float(np.mean(d2))
        complexity[voxel_id] = np.sqrt(max(spread, 0.0))

    # Sparse voxels receive higher scores. Log scaling limits the
    # effect of isolated outliers and extremely dense voxels.
    sparsity = 1.0 / np.sqrt(voxel_sizes.astype(np.float64))
    sparsity = (sparsity - sparsity.min()) / (
        sparsity.max() - sparsity.min() + 1e-12
    )
    complexity = complexity.astype(np.float64)
    complexity = (complexity - complexity.min()) / (
        complexity.max() - complexity.min() + 1e-12
    )
    score = 0.65 * sparsity + 0.35 * complexity

    rng = np.random.default_rng(random_state)

    if target <= voxel_count:
        chosen = _select_spatially_spread(centers, score, target)
        selected = representatives[chosen]
    else:
        selected = list(representatives)
        remaining = target - voxel_count

        valid = np.flatnonzero(secondary >= 0)
        if remaining > 0 and len(valid):
            ranked = valid[np.argsort(score[valid])[::-1]]
            take = min(remaining, len(ranked))
            selected.extend(secondary[ranked[:take]])
            remaining -= take

        if remaining > 0:
            already = np.zeros(n, dtype=bool)
            already[np.asarray(selected, dtype=np.int64)] = True
            available = np.flatnonzero(~already)

            if len(available):
                point_scores = score[inverse[available]]
                point_scores += rng.random(len(available)) * 1e-7
                ranked = available[np.argsort(point_scores)[::-1]]
                take = min(remaining, len(ranked))
                selected.extend(ranked[:take])

        selected = np.unique(np.asarray(selected, dtype=np.int64))

        if len(selected) > target:
            selected = selected[:target]
        elif len(selected) < target:
            already = np.zeros(n, dtype=bool)
            already[selected] = True
            available = np.flatnonzero(~already)
            selected = np.concatenate(
                (selected, available[:target - len(selected)])
            )

    result = xyz[selected]

    if verbose:
        print(f"Original points: {n}")
        print(f"Occupied voxels: {voxel_count}")
        print(f"Target points: {target}")
        print(f"Output points: {len(result)}")
        print(f"Total wall time: {perf_counter() - start:.3f} s")

    return result.astype(np.float32)
