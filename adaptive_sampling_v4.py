import numpy as np
from time import perf_counter

def _farthest_point_sampling_fast(points, count):
    """Fast bounded Farthest Point Sampling."""
    n = len(points)
    if count >= n:
        return np.arange(n, dtype=np.int64)

    selected = np.empty(count, dtype=np.int64)
    min_dist = np.full(n, np.inf, dtype=np.float32)

    # Start near the centroid
    centroid = points.mean(axis=0)
    selected[0] = np.argmin(np.sum((points - centroid) ** 2, axis=1))

    for i in range(1, count):
        current = selected[i - 1]
        dist = np.sum((points - points[current]) ** 2, axis=1)
        np.minimum(min_dist, dist, out=min_dist)
        selected[i] = np.argmax(min_dist)

    return selected

def adaptive_sample_v4(xyz, target_points=9416, base_voxel_size=0.4):
    """
    V4 Adaptive Sampling: Distance-Banded Voxel Variance.
    Combines V3 distance allocation with ultra-fast Z-variance complexity 
    to preserve small vertical objects (pedestrians) without KD-tree overhead.
    """
    t0 = perf_counter()
    xyz = np.asarray(xyz, dtype=np.float32)
    n = len(xyz)

    if target_points >= n:
        return xyz.copy()
    if target_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    # 1. Distance Band Budget Allocation (Borrowed from V3)
    ranges = np.linalg.norm(xyz[:, :2], axis=1) # 2D range often better for LiDAR
    edges = [0, 10, 20, 40, np.inf]
    
    # Over-index on near-mid range where pedestrians usually lie
    band_weights = np.array([0.30, 0.35, 0.22, 0.13]) 
    band_ids = np.digitize(ranges, edges[1:-1])
    counts = np.bincount(band_ids, minlength=4)
    active = counts > 0

    weights = band_weights * active
    weights /= weights.sum()
    raw = weights * target_points
    budgets = np.minimum(np.floor(raw).astype(int), counts)

    remaining = target_points - budgets.sum()
    while remaining > 0:
        eligible = np.flatnonzero(budgets < counts)
        if len(eligible) == 0:
            break
        deficits = raw[eligible] - budgets[eligible]
        order = eligible[np.argsort(deficits)[::-1]]
        take = min(remaining, len(order))
        budgets[order[:take]] += 1
        remaining -= take

    selected_indices = []

    # 2. Process Each Band Independently
    for band in range(4):
        budget = int(budgets[band])
        band_mask = (band_ids == band)
        band_idx = np.flatnonzero(band_mask)

        if budget <= 0 or len(band_idx) == 0:
            continue
        if budget >= len(band_idx):
            selected_indices.extend(band_idx.tolist())
            continue

        band_pts = xyz[band_idx]
        
        # Scale voxel size by distance band to account for density drop-off
        current_voxel_size = base_voxel_size * (1.5 ** band) 
        
        # Voxelize the band
        voxel_coords = np.floor(band_pts / current_voxel_size).astype(np.int32)
        _, inverse, voxel_counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
        num_voxels = len(voxel_counts)
        
        # Calculate fast geometric complexity: Z-variance within voxels
        # Objects like pedestrians have high vertical variance compared to flat ground
        z_vals = band_pts[:, 2]
        sum_z = np.bincount(inverse, weights=z_vals)
        sum_z2 = np.bincount(inverse, weights=z_vals**2)
        
        mean_z = sum_z / voxel_counts
        var_z = (sum_z2 / voxel_counts) - (mean_z ** 2)
        var_z = np.maximum(var_z, 0.0) # Floating point safety
        
        # Map complexity back to points
        point_complexity = var_z[inverse]
        
        if num_voxels >= budget:
            # We have more voxels than budget: keep voxels with highest complexity
            # Pick the point closest to the voxel mean in high-variance voxels
            centers = np.column_stack([
                np.bincount(inverse, weights=band_pts[:, d]) / voxel_counts for d in range(3)
            ])
            d2_to_center = np.sum((band_pts - centers[inverse]) ** 2, axis=1)
            
            # Sort by voxel variance (descending), then distance to center (ascending)
            # Add a small penalty to dense voxels to encourage spatial spread
            priority = point_complexity / (np.log(voxel_counts[inverse] + 1))
            
            order = np.lexsort((d2_to_center, -priority, inverse))
            ordered_cells = inverse[order]
            first = np.r_[True, ordered_cells[1:] != ordered_cells[:-1]]
            
            representatives = order[first]
            # Take top 'budget' voxels sorted by variance
            top_voxels = np.argsort(-var_z[ordered_cells[first]])[:budget]
            chosen_local = representatives[top_voxels]
            
        else:
            # We have fewer voxels than budget: Keep all voxel centers, FPS the rest
            centers = np.column_stack([
                np.bincount(inverse, weights=band_pts[:, d]) / voxel_counts for d in range(3)
            ])
            d2_to_center = np.sum((band_pts - centers[inverse]) ** 2, axis=1)
            order = np.lexsort((d2_to_center, inverse))
            ordered_cells = inverse[order]
            first = np.r_[True, ordered_cells[1:] != ordered_cells[:-1]]
            
            chosen_local = order[first].tolist()
            
            # Fill the remaining budget using bounded FPS to ensure widespread coverage
            remaining_budget = budget - len(chosen_local)
            mask = np.ones(len(band_pts), dtype=bool)
            mask[chosen_local] = False
            remaining_pool = np.flatnonzero(mask)
            
            if len(remaining_pool) > 0:
                fps_pool_idx = remaining_pool
                if len(fps_pool_idx) > remaining_budget * 5:
                    # Bound pool for speed
                    fps_pool_idx = np.random.choice(fps_pool_idx, remaining_budget * 5, replace=False)
                    
                fps_local = _farthest_point_sampling_fast(band_pts[fps_pool_idx], remaining_budget)
                chosen_local.extend(fps_pool_idx[fps_local])

        selected_indices.extend(band_idx[chosen_local].tolist())

    selected = np.array(selected_indices, dtype=np.int64)

    # 3. Final Budget Guard
    if len(selected) > target_points:
        selected = selected[:target_points]
    elif len(selected) < target_points:
        remaining_ids = np.setdiff1d(np.arange(n), selected, assume_unique=False)
        extra = remaining_ids[:target_points - len(selected)]
        selected = np.concatenate([selected, extra])

    print(f"Adaptive V4 Time: {perf_counter() - t0:.3f} s")
    return xyz[selected]