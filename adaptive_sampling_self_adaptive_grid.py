"""
Self-adaptive grid point-cloud downsampling inspired by:
Zu et al., "Research on self-adaptive grid point cloud down-sampling method
based on plane fitting and Mahalanobis distance Gaussian weighting",
Neurocomputing 634 (2025), 129746.

This is an implementation adapted for KITTI-style XYZ point clouds and a
fixed output budget. It is NOT a bit-for-bit reproduction: the paper leaves
some implementation details dataset-dependent, and this version uses a
deterministic bounded RANSAC and a final budget-control stage.

Input:  xyz array of shape (N, 3), in metres.
Output: sampled XYZ array of shape (min(target_points, N), 3).

Dependencies: numpy, scipy
"""

from __future__ import annotations
import time
import numpy as np
from scipy.spatial import cKDTree


def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def _boundary_scores(points: np.ndarray, k: int = 12) -> np.ndarray:
    """Paper Eq. (5): magnitude of the mean unit vectors to K neighbours."""
    n = len(points)
    if n < 4:
        return np.ones(n, dtype=np.float64)

    k_eff = min(k + 1, n)
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k_eff, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    # first neighbor is normally the point itself; exclude it
    idx = idx[:, 1:]
    vec = points[idx] - points[:, None, :]
    norms = np.linalg.norm(vec, axis=2, keepdims=True)
    unit = vec / np.maximum(norms, 1e-12)
    return np.linalg.norm(unit.sum(axis=1), axis=1) / max(unit.shape[1], 1)


def _weighted_normal(
    points: np.ndarray,
    point_index: int,
    k: int = 20,
    inv_cov=None,
):
    """
    Mahalanobis-neighbourhood + Gaussian-weighted local covariance.
    Reuses inverse covariance when supplied.
    Returns the eigenvector of the smallest eigenvalue.
    """
    n = len(points)
    if n < 3:
        return None

    k_eff = min(k, n)
    p = points[point_index]

    # Calculate covariance only if it has not already been provided.
    if inv_cov is None:
        centered = points - points.mean(axis=0, keepdims=True)
        cov_all = centered.T @ centered / max(n, 1)
        cov_all += np.eye(3) * 1e-8
        inv_cov = np.linalg.pinv(cov_all)

    delta = points - p
    mahal_sq = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)

    # Partial selection instead of sorting every distance.
    chosen = np.argpartition(mahal_sq, k_eff - 1)[:k_eff]

    local = points[chosen]
    local_delta = local - p
    md2 = np.maximum(mahal_sq[chosen], 0.0)

    bandwidth = float(np.sqrt(np.mean(md2)) + 1e-8)
    weights = np.exp(-0.5 * md2 / (bandwidth * bandwidth))
    weights /= weights.sum() + 1e-12

    mean = np.sum(local * weights[:, None], axis=0)
    centered_local = local - mean
    weighted_cov = (centered_local * weights[:, None]).T @ centered_local

    vals, vecs = np.linalg.eigh(weighted_cov)
    return _unit(vecs[:, np.argmin(vals)])


def _ransac_plane(points: np.ndarray, iterations: int = 40):
    """Small deterministic RANSAC plane fit. Returns normal and inlier ratio."""
    n = len(points)
    if n < 3:
        return None, 0.0, np.inf

    rng = np.random.default_rng(0)
    scale = np.linalg.norm(np.ptp(points, axis=0)) + 1e-12
    threshold = max(0.02, 0.02 * scale)
    best_count = 0
    best_normal = None
    best_d = 0.0

    for _ in range(min(iterations, max(10, n * 2))):
        ids = rng.choice(n, size=3, replace=False)
        a, b, c = points[ids]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -float(normal @ a)
        distances = np.abs(points @ normal + d)
        inliers = distances <= threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d

    if best_normal is None or best_count < 3:
        return None, 0.0, np.inf

    inliers = np.abs(points @ best_normal + best_d) <= threshold
    selected = points[inliers]
    center = selected.mean(axis=0)
    _, _, vt = np.linalg.svd(selected - center, full_matrices=False)
    normal = _unit(vt[-1])
    distances = np.abs((points - center) @ normal)
    flatness = float(np.max(distances) - np.min(distances))
    return normal, float(inliers.mean()), flatness


def _voxel_groups(points: np.ndarray, origin: np.ndarray, cell_size: float):
    keys = np.floor((points - origin) / cell_size).astype(np.int64)
    groups = {}
    for i, key in enumerate(map(tuple, keys)):
        groups.setdefault(key, []).append(i)
    return groups


def _farthest_point_indices(points: np.ndarray, count: int, seed_idx=None):
    """Greedy FPS; returns local indices. Intended for bounded candidate sets."""
    n = len(points)
    if count <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if count >= n:
        return np.arange(n, dtype=np.int64)

    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(seed_idx) if seed_idx is not None else 0
    min_d2 = np.sum((points - points[selected[0]]) ** 2, axis=1)

    for j in range(1, count):
        next_idx = int(np.argmax(min_d2))
        selected[j] = next_idx
        d2 = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected[:j + 1]] = -1.0
    return selected


def adaptive_sample_self_adaptive_grid(
    xyz: np.ndarray,
    target_points: int = 9416,
    voxel_size: float = 0.8,
    density_threshold: int = 24,
    boundary_k: int = 12,
    normal_k: int = 20,
    normal_angle_deg: float = 18.0,
    max_split_depth: int = 3,
    ransac_iterations: int = 15,
    random_state: int = 0,
) -> np.ndarray:
    """
    Adaptive-grid downsampling based on the paper's main stages.

    - Initial voxel grid and sparse/dense classification
    - Sparse voxels: boundary-coefficient preservation + uniform selection
    - Dense voxels: RANSAC plane fitting, recursive subdivision, weighted
      normal estimation and feature-point preservation
    - Final fixed-budget selection/fill using FPS

    The fixed-budget stage is a project adaptation. Labels are never used.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("xyz must have shape (N, >=3)")
    xyz = xyz[:, :3]
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    n = len(xyz)
    if n == 0:
        return xyz
    target = min(max(int(target_points), 1), n)
    if target == n:
        return xyz.copy()

    origin = xyz.min(axis=0)
    t0 = time.perf_counter()
    groups = _voxel_groups(xyz, origin, float(voxel_size))
    voxel_time = time.perf_counter() - t0
    rng = np.random.default_rng(random_state)
    candidate_set = set()
    sparse_indices = []
    sparse_voxels = []
    dense_voxels = []

    for ids_list in groups.values():
        ids = np.asarray(ids_list, dtype=np.int64)
        pts = xyz[ids]
        if len(ids) < density_threshold:
            sparse_indices.extend(ids.tolist())
            sparse_voxels.append(ids)
        else:
            dense_voxels.append(ids)

    print(f"Total voxels: {len(groups)}")
    print(f"Sparse voxels: {len(sparse_voxels)}")
    print(f"Sparse points: {len(sparse_indices)}")
    print(f"Dense voxels: {len(dense_voxels)}")
    print(
        f"Points in dense voxels: "
        f"{sum(len(ids) for ids in dense_voxels)}"
    )

    sparse_start = time.perf_counter()
    # Process each sparse voxel independently so boundary scores never mix
    # unrelated points from distant voxels. Keep a representative from every
    # sparse voxel, then optionally retain its strongest boundary point.
    sparse_boundary_count = 0
    for ids in sparse_voxels:
        pts = xyz[ids]
        center = pts.mean(axis=0)
        representative_local = int(np.argmin(np.sum((pts - center) ** 2, axis=1)))
        candidate_set.add(int(ids[representative_local]))

        if len(ids) >= 4:
            scores = _boundary_scores(pts, boundary_k)
            order = np.argsort(scores)
            best_local = int(order[-1])
            median_score = float(np.median(scores))
            best_score = float(scores[best_local])
            # Add a boundary point only when its score is meaningfully above
            # the voxel's typical score; avoid redundant candidates otherwise.
            if best_local != representative_local and best_score > median_score + 0.10:
                candidate_set.add(int(ids[best_local]))
                sparse_boundary_count += 1

    sparse_time = time.perf_counter() - sparse_start
    print(f"Sparse voxel representatives: {len(sparse_voxels)}")
    print(f"Additional sparse boundary candidates: {sparse_boundary_count}")
    cos_threshold = np.cos(np.deg2rad(normal_angle_deg))

    ransac_time = 0.0
    normal_time = 0.0
    ransac_calls = 0
    normal_calls = 0
    
    def process_dense(ids: np.ndarray, depth: int):
        
        nonlocal ransac_time, normal_time, ransac_calls, normal_calls
        
        pts = xyz[ids]
        if len(ids) == 0:
            return

        t = time.perf_counter()
        normal, inlier_ratio, flatness = _ransac_plane(pts, ransac_iterations)
        ransac_time += time.perf_counter() - t
        ransac_calls += 1   
        scale = float(np.linalg.norm(np.ptp(pts, axis=0)) + 1e-12)
        # Planarity tolerance is scale-aware; this is a practical adaptation.
        plane_passes = normal is not None and inlier_ratio >= 0.80 and flatness <= max(0.04, 0.04 * scale)

        if not plane_passes and depth < max_split_depth and len(ids) >= 16:
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
            # Degenerate / noisy region: keep a spatially representative subset.
            candidate_set.add(int(ids[np.argmax(np.linalg.norm(pts - pts.mean(axis=0), axis=1))]))
            return

        # Estimate weighted normals on a bounded set of representative points.
        # Points with normals differing from the fitted plane are features.
        if len(ids) > 180:
            probes = np.linspace(0, len(ids) - 1, 180, dtype=np.int64)
        else:
            probes = np.arange(len(ids), dtype=np.int64)
            
        # Precompute inverse covariance once for this dense region.
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov_all = centered.T @ centered / max(len(pts), 1)
        cov_all += np.eye(3) * 1e-8
        inv_cov = np.linalg.pinv(cov_all)

        for local_i in probes:
            t = time.perf_counter()
            local_normal = _weighted_normal(pts, int(local_i), normal_k, inv_cov=inv_cov)
            normal_time += time.perf_counter() - t
            normal_calls += 1
            if local_normal is None:
                continue
            # Plane normal sign is ambiguous; compare absolute cosine.
            cosine = abs(float(np.dot(local_normal, normal)))
            if cosine < cos_threshold:
                candidate_set.add(int(ids[local_i]))

        # Preserve one representative from every accepted dense grid.
        candidate_set.add(int(ids[len(ids) // 2]))

    dense_start = time.perf_counter()
    for ids in dense_voxels:
        process_dense(ids, 0)

    dense_time = time.perf_counter() - dense_start

    print(f"RANSAC total: {ransac_time:.3f} s ({ransac_calls} calls)")
    print(f"Weighted normals total: {normal_time:.3f} s ({normal_calls} calls)")
    print(
        f"Other dense-stage time: "
        f"{dense_time - ransac_time - normal_time:.3f} s"
    )

    selection_start = time.perf_counter()
    candidates = np.fromiter(candidate_set, dtype=np.int64)

    if len(candidates) > 0:
        sparse_set = set(sparse_indices)
        sparse_candidate_count = sum(int(idx) in sparse_set for idx in candidates)
        dense_candidate_count = len(candidates) - sparse_candidate_count

        print(f"Sparse points retained as candidates: {sparse_candidate_count}")
        print(f"Dense points retained as candidates: {dense_candidate_count}")
        print(
            f"Sparse candidate ratio: "
            f"{sparse_candidate_count / max(len(sparse_indices), 1):.4f}"
        )
        dense_point_count = sum(len(ids) for ids in dense_voxels)
        print(
            f"Dense candidate ratio: "
            f"{dense_candidate_count / max(dense_point_count, 1):.4f}"
        )

    print(f"Candidate count before budget selection: {len(candidates)}")
    print(f"Target point budget: {target}")
    print(f"Original point count: {n}")

    if len(candidates) == 0:
        candidates = np.arange(n, dtype=np.int64)

    candidate_tree = cKDTree(xyz[candidates])
    candidate_distances, _ = candidate_tree.query(xyz, k=1, workers=-1)
    print(
        f"Candidate coverage: mean={candidate_distances.mean():.4f} m, "
        f"p95={np.percentile(candidate_distances, 95):.4f} m"
    )

    # If candidates exceed budget, FPS selects a spatially spread subset.
    if len(candidates) >= target:
        cand_xyz = xyz[candidates]
        # Bound FPS cost if feature set is unusually large.
        if len(candidates) > max(target * 8, 50000):
            # Deterministically pre-thin candidate pool, then FPS.
            pre_ids = np.linspace(0, len(candidates) - 1, max(target * 8, 50000), dtype=np.int64)
            candidates = candidates[pre_ids]
            cand_xyz = xyz[candidates]
        chosen_local = _farthest_point_indices(cand_xyz, target)
        result = xyz[candidates[chosen_local]].astype(np.float32)

        selection_time = time.perf_counter() - selection_start

        print(f"Voxel grouping:        {voxel_time:.3f} s")
        print(f"Sparse-grid scoring:   {sparse_time:.3f} s")
        print(f"Dense-grid processing: {dense_time:.3f} s")
        print(f"Final selection:       {selection_time:.3f} s")
        print(
            f"Total measured:        "
            f"{voxel_time + sparse_time + dense_time + selection_time:.3f} s"
        )

        return result

    # Too few candidates: fill from all original points using FPS seeded by candidates.
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[candidates] = False
    remaining_ids = np.flatnonzero(remaining_mask)
    need = target - len(candidates)
    if need <= 0:
        return xyz[candidates[:target]].astype(np.float32)

    # FPS on a bounded pool to avoid quadratic work on the full KITTI frame.
    # Include all current candidates and a deterministic spatially spread pool
    # of remaining points; then use FPS only to choose the missing budget.
    pool_limit = min(len(remaining_ids), max(8 * target, 50000))
    if len(remaining_ids) > pool_limit:
        pool_rng = np.random.default_rng(random_state)
        pool_ids = np.sort(pool_rng.choice(remaining_ids, size=pool_limit, replace=False))
    else:
        pool_ids = remaining_ids

    pool_xyz = xyz[pool_ids]
    fill_count = min(need, len(pool_ids))
    if fill_count == len(pool_ids):
        fill_ids = pool_ids
    else:
        fill_local = _farthest_point_indices(pool_xyz, fill_count)
        fill_ids = pool_ids[fill_local]

    selected = list(candidates.tolist())
    selected_set = set(selected)
    for idx in fill_ids:
        i = int(idx)
        if i not in selected_set:
            selected.append(i)
            selected_set.add(i)
            if len(selected) >= target:
                break

    # In the unlikely event candidate overlap leaves the budget short, fill
    # from remaining indices without replacement.
    if len(selected) < target:
        leftovers = np.setdiff1d(np.arange(n), np.asarray(selected, dtype=np.int64), assume_unique=False)
        selected.extend(leftovers[:target - len(selected)].tolist())
    
    result = xyz[np.asarray(selected[:target], dtype=np.int64)].astype(np.float32)
    selection_time = time.perf_counter() - selection_start

    print(f"Voxel grouping:       {voxel_time:.3f} s")
    print(f"Sparse-grid scoring:  {sparse_time:.3f} s")
    print(f"Dense-grid processing:{dense_time:.3f} s")
    print(f"Final selection:     {selection_time:.3f} s")
    print(f"Total measured:      {voxel_time + sparse_time + dense_time + selection_time:.3f} s")

    return result



# Alias with the same simple calling style used by the project's other samplers.
def adaptive_sample(xyz: np.ndarray, target_points: int = 9416, **kwargs) -> np.ndarray:
    return adaptive_sample_self_adaptive_grid(xyz, target_points=target_points, **kwargs)
