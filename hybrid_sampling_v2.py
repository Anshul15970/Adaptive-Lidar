
import time
import numpy as np

from adaptive_sampling_self_adaptive_grid import (
    _boundary_scores,
    _ransac_plane,
    _weighted_normal,
    _voxel_groups,
    _farthest_point_indices,
)


def _allocate_budget(counts, weights, target):
    """Allocate an exact point budget across nonempty distance bands."""
    counts = np.asarray(counts, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    active = counts > 0
    if not np.any(active):
        return np.zeros_like(counts)

    weights = weights * active
    weights /= weights.sum()

    raw = weights * target
    budgets = np.minimum(np.floor(raw).astype(np.int64), counts)

    while budgets.sum() < target:
        eligible = np.flatnonzero(budgets < counts)
        if len(eligible) == 0:
            break
        deficits = raw[eligible] - budgets[eligible]
        budgets[eligible[np.argmax(deficits)]] += 1

    return budgets


def _bounded_fps(xyz, ids, count, pool_limit=18000):
    """FPS on a bounded candidate pool, returning original point indices."""
    ids = np.asarray(ids, dtype=np.int64)

    if count <= 0 or len(ids) == 0:
        return np.empty(0, dtype=np.int64)

    if len(ids) <= count:
        return ids.copy()

    # Reduce very large candidate pools using spatial cells.
    if len(ids) > pool_limit:
        pts = xyz[ids]
        span = np.ptp(pts, axis=0)
        side = max((np.prod(span + 1e-6) / pool_limit) ** (1 / 3), 0.05)

        for _ in range(12):
            keys = np.floor((pts - pts.min(axis=0)) / side).astype(np.int32)
            _, inverse = np.unique(keys, axis=0, return_inverse=True)
            cell_count = int(inverse.max()) + 1

            if cell_count > pool_limit:
                side *= 1.25
            elif cell_count < pool_limit * 0.65:
                side *= 0.85
            else:
                break

        # Select the point nearest each occupied cell's centroid.
        cell_counts = np.bincount(inverse)
        centers = np.column_stack([
            np.bincount(inverse, weights=pts[:, d]) / cell_counts
            for d in range(3)
        ])
        d2 = np.sum((pts - centers[inverse]) ** 2, axis=1)

        order = np.lexsort((d2, inverse))
        ordered_cells = inverse[order]
        first = np.r_[True, ordered_cells[1:] != ordered_cells[:-1]]
        ids = ids[order[first]]

    if len(ids) <= count:
        return ids

    local = _farthest_point_indices(xyz[ids], count)
    return ids[local]


def hybrid_sample_v2(
    xyz,
    target_points=9416,
    voxel_size=0.8,
    density_threshold=24,
    boundary_k=12,
    normal_k=12,
    normal_angle_deg=18.0,
    max_split_depth=2,
    ransac_iterations=3,
    random_state=0,
):
    """
    Hybrid geometric sampler:
      - self-adaptive sparse/dense voxel feature candidates
      - V3-style distance-band budget allocation
      - bounded, band-wise spatial selection

    No semantic labels are used.
    """
    start = time.perf_counter()

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("xyz must have shape (N, >=3)")

    xyz = xyz[:, :3]
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]

    n = len(xyz)
    target = min(max(int(target_points), 1), n)

    if n == 0 or target == n:
        return xyz.copy()

    # V3-style distance bands and allocation proportions.
    ranges = np.linalg.norm(xyz, axis=1)
    edges = np.array([0.0, 10.0, 20.0, 40.0, np.inf])
    band_ids = np.digitize(ranges, edges[1:-1])
    band_counts = np.bincount(band_ids, minlength=4)
    budgets = _allocate_budget(
        band_counts, [0.35, 0.30, 0.22, 0.13], target
    )

    # Initial adaptive voxel grid.
    origin = xyz.min(axis=0)
    groups = _voxel_groups(xyz, origin, voxel_size)

    candidate_set = set()
    sparse_voxels = []
    dense_voxels = []

    for ids_list in groups.values():
        ids = np.asarray(ids_list, dtype=np.int64)
        if len(ids) < density_threshold:
            sparse_voxels.append(ids)
        else:
            dense_voxels.append(ids)

    # Sparse voxels: representative plus strong boundary point.
    for ids in sparse_voxels:
        pts = xyz[ids]
        center = pts.mean(axis=0)
        representative = int(
            np.argmin(np.sum((pts - center) ** 2, axis=1))
        )
        candidate_set.add(int(ids[representative]))

        if len(ids) >= 4:
            scores = _boundary_scores(pts, boundary_k)
            best = int(np.argmax(scores))
            if (
                best != representative
                and scores[best] > np.median(scores) + 0.10
            ):
                candidate_set.add(int(ids[best]))

    cos_threshold = np.cos(np.deg2rad(normal_angle_deg))

    # Dense voxels: plane fitting, recursive splitting and feature normals.
    def process_dense(ids, depth):
        pts = xyz[ids]
        if len(ids) == 0:
            return

        normal, inlier_ratio, flatness = _ransac_plane(
            pts, iterations=ransac_iterations
        )
        scale = float(np.linalg.norm(np.ptp(pts, axis=0)) + 1e-12)

        plane_passes = (
            normal is not None
            and inlier_ratio >= 0.80
            and flatness <= max(0.04, 0.04 * scale)
        )

        if (
            not plane_passes
            and depth < max_split_depth
            and len(ids) >= 16
        ):
            center = pts.mean(axis=0)
            bits = (pts >= center).astype(np.int8)
            codes = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]

            split_any = False
            for code in range(8):
                sub = ids[codes == code]
                if len(sub) >= 3:
                    split_any = True
                    process_dense(sub, depth + 1)

            if split_any:
                return

        if normal is None:
            candidate_set.add(
                int(ids[np.argmax(
                    np.linalg.norm(pts - pts.mean(axis=0), axis=1)
                )])
            )
            return

        # Bound normal estimation to at most 60 probes per region.
        probe_count = min(len(ids), 60)
        probes = np.linspace(
            0, len(ids) - 1, probe_count, dtype=np.int64
        )

        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(pts), 1)
        inv_cov = np.linalg.pinv(cov + np.eye(3) * 1e-8)

        for local_i in probes:
            local_normal = _weighted_normal(
                pts, int(local_i), normal_k, inv_cov=inv_cov
            )
            if local_normal is None:
                continue

            cosine = abs(float(np.dot(local_normal, normal)))
            if cosine < cos_threshold:
                candidate_set.add(int(ids[local_i]))

        # Preserve a point from each accepted dense region.
        center = pts.mean(axis=0)
        candidate_set.add(
            int(ids[np.argmin(np.sum((pts - center) ** 2, axis=1))])
        )

    for ids in dense_voxels:
        process_dense(ids, 0)

    candidates = np.fromiter(candidate_set, dtype=np.int64)
    if len(candidates) == 0:
        candidates = np.arange(n, dtype=np.int64)

    # One final selection stage, independently per distance band.
    selected_parts = []

    for band in range(4):
        band_budget = int(budgets[band])
        if band_budget <= 0:
            continue

        candidate_ids = candidates[band_ids[candidates] == band]

        if len(candidate_ids) >= band_budget:
            chosen = _bounded_fps(xyz, candidate_ids, band_budget)
        else:
            # Keep all feature candidates, then fill from points in this band.
            chosen = candidate_ids.copy()
            remaining = band_budget - len(chosen)

            band_pool = np.flatnonzero(band_ids == band)
            used = np.zeros(n, dtype=bool)
            used[chosen] = True
            fill_pool = band_pool[~used[band_pool]]

            fill = _bounded_fps(xyz, fill_pool, remaining)
            chosen = np.concatenate((chosen, fill))

        selected_parts.append(chosen)

    selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)

    # Guard against rare band/candidate shortages or duplicate indices.
    selected = np.unique(selected)

    if len(selected) < target:
        used = np.zeros(n, dtype=bool)
        used[selected] = True
        remaining_ids = np.flatnonzero(~used)
        extra = _bounded_fps(
            xyz, remaining_ids, target - len(selected)
        )
        selected = np.concatenate((selected, extra))

    elif len(selected) > target:
        selected = _bounded_fps(xyz, selected, target)

    result = xyz[selected[:target]].astype(np.float32)

    print(f"Original points: {n}")
    print(f"Candidate points: {len(candidates)}")
    print(f"Target points: {target}")
    print(f"Output points: {len(result)}")
    print(f"Total wall time: {time.perf_counter() - start:.3f} s")

    return result
