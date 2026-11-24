
import numpy as np


def adaptive_sample_v3(xyz, target_points=9416):
    """
    Distance-aware voxel sampling.
    Uses geometry and distance only; no labels.
    Returns exactly target_points when feasible.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    n = len(xyz)

    if target_points >= n:
        return xyz.copy()
    if target_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    ranges = np.linalg.norm(xyz, axis=1)
    edges = [0, 10, 20, 40, np.inf]

    # Allocate budget across distance bands.
    band_weights = np.array([0.35, 0.30, 0.22, 0.13])
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

    selected = []

    # Smaller voxels near the sensor, larger voxels farther away.
    for band in range(4):
        ids = np.flatnonzero(band_ids == band)
        budget = int(budgets[band])

        if budget == 0 or len(ids) == 0:
            continue
        if budget >= len(ids):
            selected.extend(ids.tolist())
            continue

        points = xyz[ids]
        band_range = ranges[ids]
        mean_range = float(np.mean(band_range))

        # Initial voxel size scales with distance and point budget.
        voxel_size = max(0.15, min(2.0, mean_range * 0.025))

        # Adjust voxel size to get at least the band budget of cells.
        for _ in range(14):
            cells = np.floor(points / voxel_size).astype(np.int32)
            _, inverse = np.unique(
                cells, axis=0, return_inverse=True
            )
            cell_count = inverse.max() + 1

            if cell_count > budget:
                voxel_size *= 1.12
            else:
                voxel_size /= 1.12

        cells = np.floor(points / voxel_size).astype(np.int32)
        _, inverse = np.unique(
            cells, axis=0, return_inverse=True
        )
        cell_count = inverse.max() + 1

        # Find the point closest to each occupied voxel's centroid.
        cell_counts = np.bincount(inverse, minlength=cell_count)
        centers = np.column_stack([
            np.bincount(
                inverse, weights=points[:, d],
                minlength=cell_count
            ) / cell_counts
            for d in range(3)
        ])

        squared_distance = np.sum(
            (points - centers[inverse]) ** 2, axis=1
        )

        order = np.lexsort((squared_distance, inverse))
        ordered_cells = inverse[order]
        first = np.r_[
            True, ordered_cells[1:] != ordered_cells[:-1]
        ]
        representatives = order[first]

        if len(representatives) > budget:
            # Keep spatially distributed voxel representatives.
            candidates = points[representatives]
            chosen = [int(np.argmin(
                np.sum((candidates - candidates.mean(axis=0)) ** 2,
                       axis=1)
            ))]

            min_dist = np.sum(
                (candidates - candidates[chosen[0]]) ** 2, axis=1
            )

            while len(chosen) < budget:
                min_dist[chosen] = -1
                nxt = int(np.argmax(min_dist))
                chosen.append(nxt)
                dist = np.sum(
                    (candidates - candidates[nxt]) ** 2, axis=1
                )
                min_dist = np.minimum(min_dist, dist)

            representatives = representatives[np.asarray(chosen)]

        elif len(representatives) < budget:
            used = np.zeros(len(points), dtype=bool)
            used[representatives] = True
            leftover = np.flatnonzero(~used)
            needed = budget - len(representatives)

            extra = leftover[
                np.argsort(squared_distance[leftover])[:needed]
            ]
            representatives = np.concatenate(
                [representatives, extra]
            )

        selected.extend(ids[representatives].tolist())

    selected = np.asarray(selected, dtype=int)

    # Guard against rounding or band-allocation edge cases.
    if len(selected) > target_points:
        selected = selected[:target_points]
    elif len(selected) < target_points:
        remaining_ids = np.setdiff1d(
            np.arange(n), selected, assume_unique=False
        )
        extra = remaining_ids[:target_points - len(selected)]
        selected = np.concatenate([selected, extra])

    return xyz[selected]
