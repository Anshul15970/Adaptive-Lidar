
from pathlib import Path
from time import perf_counter
import cProfile
import pstats
import io

import numpy as np

from hybrid_sampling_v4 import hybrid_sample_v4

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training" / "velodyne"

files = sorted(DATA.glob("*.bin"))[:5]

if not files:
    raise FileNotFoundError(f"No KITTI point clouds found in {DATA}")

for path in files:
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    # Use the same target budget as the existing benchmark.
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    target = len(cloud.voxel_down_sample(0.8).points)

    profiler = cProfile.Profile()
    start = perf_counter()
    profiler.enable()

    result = hybrid_sample_v4(
        xyz,
        target_points=target,
        voxel_size=0.8,
        random_state=0,
    )

    profiler.disable()
    elapsed = perf_counter() - start

    print(f"\nFrame: {path.stem}")
    print(f"Input points: {len(xyz):,}")
    print(f"Target points: {target:,}")
    print(f"Output points: {len(result):,}")
    print(f"Elapsed time: {elapsed:.4f} s")

    report = io.StringIO()
    stats = pstats.Stats(profiler, stream=report).sort_stats("cumtime")
    stats.print_stats(12)
    print(report.getvalue())
