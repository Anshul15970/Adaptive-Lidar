from pathlib import Path
import numpy as np
from hybrid_sampling_v5 import hybrid_sample_v5

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/object_detection/training/velodyne"
SAMPLED = ROOT / "data/detector_sampling/hybrid_v5/velodyne"
SPLIT = Path.home() / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"

VOXEL_SIZE = 0.8
FRAME_LIMIT = 50


def trace_selection(xyz, target_points, voxel_size=0.8, random_state=0):
    xyz = np.asarray(xyz, dtype=np.float32)
    xyz = xyz[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    original_indices = np.flatnonzero(finite)
    xyz = xyz[finite]

    n = len(xyz)
    target = min(max(int(target_points), 0), n)

    if target == 0:
        return original_indices[:0], np.empty(0, dtype=np.int8)

    if target == n:
        return original_indices, np.full(n, 1, dtype=np.int8)

    keys = np.floor(xyz / voxel_size).astype(np.int32)
    unique_keys, inverse, voxel_sizes = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    voxel_count = len(unique_keys)

    order = np.argsort(inverse, kind="stable")
    boundaries = np.r_[
        0, np.flatnonzero(np.diff(inverse[order])) + 1, n
    ]

    centers = np.empty((voxel_count, 3), dtype=np.float32)
    representatives = np.empty(voxel_count, dtype=np.int64)
    secondary = np.full(voxel_count, -1, dtype=np.int64)
    complexity = np.zeros(voxel_count, dtype=np.float32)

    for voxel_id, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        ids = order[left:right]
        points = xyz[ids]
        center = points.mean(axis=0)
        centers[voxel_id] = center

        d2 = np.sum((points - center) ** 2, axis=1)
        representatives[voxel_id] = ids[int(np.argmin(d2))]

        if len(ids) > 1:
            secondary[voxel_id] = ids[int(np.argmax(d2))]

        complexity[voxel_id] = np.sqrt(max(float(np.mean(d2)), 0.0))

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

    # Stage IDs:
    # 1 = initial voxel representatives
    # 2 = secondary points
    # 3 = score-ranked remaining points
    # 4 = final fallback points
    selected = []
    stages = []

    if target <= voxel_count:
        # This is the spatial representative branch.
        # It is reproduced here for completeness.
        count = target
        scores = score.copy()
        scores -= scores.min()
        scores /= scores.max() + 1e-12

        chosen = np.empty(count, dtype=np.int64)
        used = np.zeros(voxel_count, dtype=bool)
        min_dist = np.full(voxel_count, np.inf)

        first = int(np.argmax(scores))
        chosen[0] = first
        used[first] = True

        for i in range(1, count):
            delta = centers - centers[chosen[i - 1]]
            d2 = np.einsum("ij,ij->i", delta, delta)
            np.minimum(min_dist, d2, out=min_dist)
            spread = min_dist / (min_dist.max() + 1e-12)
            priority = spread * (0.65 + 0.35 * scores)
            priority[used] = -np.inf
            nxt = int(np.argmax(priority))
            chosen[i] = nxt
            used[nxt] = True

        selected = representatives[chosen].tolist()
        stages = [1] * len(selected)

    else:
        # Stage 1: one representative from every occupied voxel.
        selected = representatives.tolist()
        stages = [1] * len(selected)
        remaining = target - voxel_count

        # Stage 2: farthest-from-centroid secondary point.
        valid = np.flatnonzero(secondary >= 0)
        if remaining > 0 and len(valid):
            ranked = valid[np.argsort(score[valid])[::-1]]
            take = min(remaining, len(ranked))
            extra = secondary[ranked[:take]].tolist()
            selected.extend(extra)
            stages.extend([2] * len(extra))
            remaining -= take

        # Stage 3: remaining points, ranked by their voxel score.
        if remaining > 0:
            already = np.zeros(n, dtype=bool)
            already[np.asarray(selected, dtype=np.int64)] = True
            available = np.flatnonzero(~already)

            if len(available):
                point_scores = score[inverse[available]].copy()
                point_scores += rng.random(len(available)) * 1e-7
                ranked = available[np.argsort(point_scores)[::-1]]
                take = min(remaining, len(ranked))
                extra = ranked[:take].tolist()
                selected.extend(extra)
                stages.extend([3] * len(extra))
                remaining -= take

        # Match V5's final unique/truncate/pad behaviour.
        stage_by_index = dict(zip(selected, stages))
        selected = np.unique(np.asarray(selected, dtype=np.int64))

        if len(selected) > target:
            selected = selected[:target]
        elif len(selected) < target:
            already = np.zeros(n, dtype=bool)
            already[selected] = True
            available = np.flatnonzero(~already)
            extra = available[:target - len(selected)]
            selected = np.concatenate((selected, extra))
            for idx in extra:
                stage_by_index[int(idx)] = 4

        stages = [stage_by_index[int(idx)] for idx in selected]

    return original_indices[selected], np.asarray(stages, dtype=np.int8)


def main():
    frame_ids = [
        x.strip() for x in SPLIT.read_text().splitlines() if x.strip()
    ][:FRAME_LIMIT]

    totals = np.zeros(5, dtype=np.int64)
    verified = 0

    for frame_id in frame_ids:
        path = SOURCE / f"{frame_id}.bin"
        sampled_path = SAMPLED / f"{frame_id}.bin"

        if not path.exists() or not sampled_path.exists():
            continue

        raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        exported = np.fromfile(
            sampled_path, dtype=np.float32
        ).reshape(-1, 4)

        target = len(raw) // 5
        indices, stages = trace_selection(
            raw[:, :3], target, VOXEL_SIZE, random_state=0
        )

        expected = hybrid_sample_v5(
            raw[:, :3],
            target_points=target,
            voxel_size=VOXEL_SIZE,
            random_state=0,
        )

        traced_xyz = raw[indices, :3]

        # Compare coordinate multisets, not row order. The exporter
        # maps coordinates through a dictionary, so duplicate XYZ rows
        # can have ambiguous intensity values.
        expected_rows = sorted(map(tuple, expected.tolist()))
        traced_rows = sorted(map(tuple, traced_xyz.tolist()))

        if expected_rows != traced_rows:
            raise RuntimeError(
                f"{frame_id}: traced selection does not match V5 output"
            )

        if len(indices) != len(exported):
            raise RuntimeError(
                f"{frame_id}: traced {len(indices)} points, "
                f"export contains {len(exported)}"
            )

        counts = np.bincount(stages, minlength=5)
        totals += counts
        verified += 1

        print(
            f"{frame_id}: "
            f"representatives={counts[1]}, "
            f"secondary={counts[2]}, "
            f"score_fill={counts[3]}, "
            f"fallback={counts[4]}, "
            f"total={len(indices)}"
        )

    print("\n========== VERIFIED SUMMARY ==========")
    print("Frames verified:", verified)
    print("Initial representatives:", totals[1])
    print("Secondary points:", totals[2])
    print("Score-ranked fill points:", totals[3])
    print("Fallback points:", totals[4])
    print("Total traced points:", totals[1:].sum())


if __name__ == "__main__":
    main()
