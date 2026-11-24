
from pathlib import Path
from time import perf_counter
import csv

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

from adaptive_sampling import adaptive_sample
from adaptive_sampling_v2 import adaptive_sample_v2
from adaptive_sampling_v3 import adaptive_sample_v3
from adaptive_sampling_self_adaptive_grid import (
    adaptive_sample_self_adaptive_grid,
)
from hybrid_sampling_v2 import hybrid_sample_v2


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training" / "velodyne"
OUT = ROOT / "results" / "hybrid_v2_comparison.csv"

FRAME_LIMIT = 10
VOXEL_SIZE = 0.8
NEIGHBORS = 10
TOLERANCE = 0.3
SAMPLE_SIZE = 1000


def evaluate(original, reduced):
    distances, _ = cKDTree(reduced).query(original, k=1)

    rng = np.random.default_rng(42)
    ids = rng.choice(
        len(original),
        min(SAMPLE_SIZE, len(original)),
        replace=False,
    )
    sample = original[ids]

    _, neighbors = cKDTree(original).query(
        sample, k=min(NEIGHBORS + 1, len(original))
    )
    neighbor_points = original[neighbors[:, 1:]].reshape(-1, 3)

    neighbor_distances, _ = cKDTree(reduced).query(
        neighbor_points, k=1
    )

    return {
        "mean_error_m": float(distances.mean()),
        "p95_error_m": float(np.percentile(distances, 95)),
        "neighborhood_recall": float(
            np.mean(neighbor_distances <= TOLERANCE)
        ),
    }


def main():
    files = sorted(DATA.glob("*.bin"))[:FRAME_LIMIT]
    if not files:
        raise FileNotFoundError(f"No KITTI .bin files found in {DATA}")

    rows = []

    for path in tqdm(files, desc="Hybrid V2 comparison", unit="frame"):
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]

        # Uniform baseline; measure its runtime too.
        start = perf_counter()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        uniform = np.asarray(
            cloud.voxel_down_sample(VOXEL_SIZE).points
        )
        uniform_time = perf_counter() - start
        budget = len(uniform)

        if budget == 0:
            continue

        methods = [
            ("uniform", uniform, uniform_time),
        ]

        samplers = [
            ("adaptive_v1", lambda: adaptive_sample(xyz, budget)),
            ("adaptive_v2", lambda: adaptive_sample_v2(xyz, budget)),
            ("adaptive_v3", lambda: adaptive_sample_v3(xyz, budget)),
            (
                "self_adaptive_grid",
                lambda: adaptive_sample_self_adaptive_grid(
                    xyz,
                    target_points=budget,
                    voxel_size=VOXEL_SIZE,
                    ransac_iterations=15,
                ),
            ),
            (
                "hybrid_v2",
                lambda: hybrid_sample_v2(
                    xyz,
                    target_points=budget,
                    voxel_size=VOXEL_SIZE,
                ),
            ),
        ]

        for name, sampler in samplers:
            start = perf_counter()
            reduced = sampler()
            runtime = perf_counter() - start

            if len(reduced) != budget:
                raise ValueError(
                    f"{path.name} / {name}: expected {budget} points, "
                    f"got {len(reduced)}"
                )

            metrics = evaluate(xyz, reduced)
            rows.append({
                "frame": path.stem,
                "method": name,
                "original_points": len(xyz),
                "reduced_points": len(reduced),
                "reduction_pct": 100 * (1 - len(reduced) / len(xyz)),
                "runtime_s": runtime,
                **metrics,
            })

    if not rows:
        raise RuntimeError("No frames were successfully evaluated.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nAverage metrics across evaluated frames:")
    for method in (
        "uniform",
        "adaptive_v1",
        "adaptive_v2",
        "adaptive_v3",
        "self_adaptive_grid",
        "hybrid_v2",
    ):
        subset = [r for r in rows if r["method"] == method]
        if not subset:
            continue

        print(f"\n{method}")
        for key in (
            "reduction_pct",
            "mean_error_m",
            "p95_error_m",
            "neighborhood_recall",
            "runtime_s",
        ):
            print(f"  {key}: {np.mean([r[key] for r in subset]):.4f}")

    print(f"\nFrames evaluated: {len(files)}")
    print(f"Saved results to: {OUT}")


if __name__ == "__main__":
    main()
