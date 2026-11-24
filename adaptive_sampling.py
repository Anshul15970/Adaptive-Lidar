
import numpy as np
from features import compute_features
from time import perf_counter

def adaptive_sample(xyz, target_points=9416, k=20):
    n = len(xyz)

    if target_points >= n:
        return xyz.copy()

    # 1. Compute geometric complexity
    scores = compute_features(xyz, k=k)

    # 2. Assign points to distance bands
    ranges = np.linalg.norm(xyz, axis=1)
    band_edges = [0, 10, 20, 40, np.inf]
    band_ids = np.digitize(ranges, band_edges[1:-1])

    # Reserve a minimum share for each populated band
    band_weights = np.array([0.25, 0.30, 0.25, 0.20])
    band_counts = np.bincount(band_ids, minlength=4)

    active = band_counts > 0
    weights = band_weights * active
    weights /= weights.sum()

    raw_budgets = weights * target_points
    band_budgets = np.minimum(
        np.floor(raw_budgets).astype(int), band_counts
    )

    # Distribute remaining points to bands with spare capacity
    remaining = target_points - band_budgets.sum()

    while remaining > 0:
        eligible = np.flatnonzero(band_budgets < band_counts)
        if len(eligible) == 0:
            break

        # Prefer bands furthest below their intended allocation
        deficits = raw_budgets[eligible] - band_budgets[eligible]
        order = eligible[np.argsort(deficits)[::-1]]
        take = min(remaining, len(order))
        band_budgets[order[:take]] += 1
        remaining -= take

    selected = []

    # 3. Divide each distance band into spatial regions
    for band in range(4):
        band_indices = np.flatnonzero(band_ids == band)
        budget = int(band_budgets[band])

        if budget <= 0 or len(band_indices) == 0:
            continue

        if budget >= len(band_indices):
            selected.extend(band_indices.tolist())
            continue

        band_xyz = xyz[band_indices]
        band_scores = scores[band_indices]

        # Region size grows with distance to limit region count
        mean_range = np.mean(ranges[band_indices])
        region_size = max(1.0, min(4.0, mean_range * 0.05))

        region_ids = np.floor(
            band_xyz / region_size
        ).astype(np.int32)

        _, inverse = np.unique(
            region_ids, axis=0, return_inverse=True
        )

        region_count = inverse.max() + 1
        counts = np.bincount(inverse, minlength=region_count)
        score_sum = np.bincount(
            inverse, weights=band_scores, minlength=region_count
        )
        mean_complexity = score_sum / np.maximum(counts, 1)

        # Allocate points to regions using population and complexity
        region_weights = counts * (0.5 + mean_complexity)
        allocations = region_weights / region_weights.sum() * budget

        allocations = np.minimum(allocations, counts)
        region_budgets = np.floor(allocations).astype(int)

        remaining_region = budget - region_budgets.sum()
        fractions = allocations - np.floor(allocations)

        while remaining_region > 0:
            eligible = np.flatnonzero(region_budgets < counts)
            if len(eligible) == 0:
                break

            order = eligible[
                np.argsort(fractions[eligible])[::-1]
            ]
            take = min(remaining_region, len(order))
            region_budgets[order[:take]] += 1
            fractions[order[:take]] = 0
            remaining_region -= take

        # Select spatially distributed representatives
        for region in range(region_count):
            local_budget = int(region_budgets[region])
            local = np.flatnonzero(inverse == region)

            if local_budget == 0:
                continue

            if local_budget >= len(local):
                chosen = local
            else:
                local_xyz = band_xyz[local]
                local_complexity = band_scores[local]

                
                # Start near the region centroid
                centroid = local_xyz.mean(axis=0)
                first = np.argmin(
                    np.sum((local_xyz - centroid) ** 2, axis=1)
                )

                chosen_local = [first]
                
                # Maintain minimum squared distance to selected points
                min_dist = np.sum(
                    (local_xyz - local_xyz[first]) ** 2, axis=1
                )

                for _ in range(1, local_budget):
                    
                    complexity_weight = (1.0 + 2.0 * local_complexity /
                                    (local_complexity.max() + 1e-8)
                                    )
                    priority = min_dist * complexity_weight
                    priority[chosen_local] = -1

                    nxt = np.argmax(priority)
                    chosen_local.append(nxt)

                    diff = local_xyz - local_xyz[nxt]
                    dist = np.sum(
                        (diff) ** 2, axis=1
                    )
                    min_dist = np.minimum(min_dist, dist)

                chosen = local[np.array(chosen_local)]

            selected.extend(band_indices[chosen].tolist())

    # 4. Return exactly the requested number when possible
    selected = np.array(selected, dtype=int)

    if len(selected) > target_points:
        selected = selected[:target_points]

    return xyz[selected]
