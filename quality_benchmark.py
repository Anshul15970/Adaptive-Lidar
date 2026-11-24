
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
from hybrid_sampling_v4 import hybrid_sample_v4


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training" / "velodyne"
OUT = ROOT / "results" / "quality_benchmark.csv"

FRAME_LIMIT = 20
VOXEL_SIZE = 0.8
NEIGHBORS = 10
TOLERANCE = 0.3
SAMPLE_SIZE = 1000


def evaluate(original, reduced):
    # Reconstruction error: each original point to its nearest
    # point in the reduced cloud.
    distances, _ = cKDTree(reduced).query(original, k=1)

    # Preserve the existing neighborhood-recall definition.
    rng = np.random.default_rng(42)
    ids = rng.choice(
        len(original),
        min(SAMPLE_SIZE, len(original)),
        replace=False,
    )
    sample = original[ids]

    _, neighbors = cKDTree(original).query(
        sample,
        k=min(NEIGHBORS + 1, len(original)),
    )
    neighbor_points = original[neighbors[:, 1:]].reshape(-1, 3)

    neighbor_distances, _ = cKDTree(reduced).query(
        neighbor_points,
        k=1,
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

    for path in tqdm(files, desc="Quality benchmark", unit="frame"):
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]

        # Uniform voxel baseline and its runtime.
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)

        start = perf_counter()
        uniform = np.asarray(
            cloud.voxel_down_sample(VOXEL_SIZE).points
        )
        uniform_time = perf_counter() - start

        budget = len(uniform)

        # Every method receives the same target point budget.
        methods = [
            ("uniform", uniform, uniform_time),
        ]

        start = perf_counter()
        v1 = adaptive_sample(xyz, budget)
        methods.append(("adaptive_v1", v1, perf_counter() - start))

        start = perf_counter()
        v2 = adaptive_sample_v2(xyz, budget)
        methods.append(("adaptive_v2", v2, perf_counter() - start))

        start = perf_counter()
        v3 = adaptive_sample_v3(xyz, budget)
        methods.append(("adaptive_v3", v3, perf_counter() - start))

        start = perf_counter()
        self_adaptive = adaptive_sample_self_adaptive_grid(
            xyz,
            target_points=budget,
            voxel_size=VOXEL_SIZE,
            ransac_iterations=15,
        )
        methods.append((
            "self_adaptive_grid",
            self_adaptive,
            perf_counter() - start,
        ))

        start = perf_counter()
        hybrid_v4 = hybrid_sample_v4(
            xyz,
            target_points=budget,
            voxel_size=VOXEL_SIZE,
            random_state=0,
        )
        methods.append((
            "hybrid_v4",
            hybrid_v4,
            perf_counter() - start,
        ))

        for name, reduced, runtime in methods:
            if len(reduced) == 0:
                raise ValueError(
                    f"{name} returned zero points for frame {path.stem}"
                )

            metrics = evaluate(xyz, reduced)

            rows.append({
                "frame": path.stem,
                "method": name,
                "original_points": len(xyz),
                "target_points": budget,
                "reduced_points": len(reduced),
                "reduction_pct": (
                    100 * (1 - len(reduced) / len(xyz))
                ),
                "runtime_s": runtime,
                **metrics,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    method_names = [
        "uniform",
        "adaptive_v1",
        "adaptive_v2",
        "adaptive_v3",
        "self_adaptive_grid",
        "hybrid_v4",
    ]

    print(f"\nFrames evaluated: {len(files)}")
    print(f"Voxel size: {VOXEL_SIZE}")
    print(f"Neighborhood tolerance: {TOLERANCE} m")
    print("\nAverage quality metrics across frames:")

    for method in method_names:
        subset = [r for r in rows if r["method"] == method]

        print(f"\n{method}")
        for key in (
            "reduced_points",
            "reduction_pct",
            "mean_error_m",
            "p95_error_m",
            "neighborhood_recall",
            "runtime_s",
        ):
            values = [r[key] for r in subset]
            print(f"  {key}: {np.mean(values):.4f}")

    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
