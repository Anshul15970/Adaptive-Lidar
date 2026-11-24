
import time
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree

from adaptive_sampling import adaptive_sample

DATA_DIR = Path("data/raw/velodyne_points/data")
OUTPUT = Path("results")
OUTPUT.mkdir(exist_ok=True)

files = sorted(DATA_DIR.glob("*.bin"))[:10]
if not files:
    raise FileNotFoundError(f"No KITTI frames found in {DATA_DIR}")

def mean_and_p95(original, reduced):
    distances, _ = cKDTree(reduced).query(original, k=1)
    return distances.mean(), np.percentile(distances, 95)

def neighborhood_recall(original, reduced, k=10, tolerance=0.3):
    rng = np.random.default_rng(42)
    ids = rng.choice(len(original), min(1000, len(original)),
                     replace=False)
    sample = original[ids]

    _, neighbors = cKDTree(original).query(sample, k=k + 1)
    neighbor_points = original[neighbors[:, 1:]].reshape(-1, 3)

    distances, _ = cKDTree(reduced).query(neighbor_points, k=1)
    return np.mean(distances <= tolerance)

rows = []

for file in files:
    points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    uniform = np.asarray(pcd.voxel_down_sample(0.8).points)

    start = time.perf_counter()
    adaptive = adaptive_sample(xyz, target_points=len(uniform))
    adaptive_seconds = time.perf_counter() - start

    u_mean, u_p95 = mean_and_p95(xyz, uniform)
    a_mean, a_p95 = mean_and_p95(xyz, adaptive)

    u_recall = neighborhood_recall(xyz, uniform)
    a_recall = neighborhood_recall(xyz, adaptive)

    rows.append({
        "frame": file.name,
        "original_points": len(xyz),
        "uniform_points": len(uniform),
        "adaptive_points": len(adaptive),
        "uniform_mean_error_m": u_mean,
        "adaptive_mean_error_m": a_mean,
        "uniform_p95_error_m": u_p95,
        "adaptive_p95_error_m": a_p95,
        "uniform_neighborhood_recall": u_recall,
        "adaptive_neighborhood_recall": a_recall,
        "adaptive_seconds": adaptive_seconds,
    })

    print(f"Completed {file.name}")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT / "benchmark.csv", index=False)

print("\nPer-frame results:")
print(df.round(4).to_string(index=False))

print("\nAverage metrics:")
print(df.select_dtypes(include="number").mean().round(4))

print(f"\nSaved to {OUTPUT / 'benchmark.csv'}")
