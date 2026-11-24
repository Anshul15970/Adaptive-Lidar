from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from adaptive_sampling_self_adaptive_grid import (
    adaptive_sample_self_adaptive_grid,
)
import open3d as o3d

DATA_DIR = Path("data/object_detection/training/velodyne")

# Replace this with the exact filename stem found in Step 2.
FRAME_NAME = "000015"

path = DATA_DIR / f"{FRAME_NAME}.bin"

if not path.exists():
    raise FileNotFoundError(f"Frame not found: {path.resolve()}")

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

print(f"Frame: {path.name}")
print(f"Original points: {len(xyz)}")

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(xyz)

uniform = np.asarray(
    cloud.voxel_down_sample(0.8).points
)

# Match the adaptive sampler's output budget.
if len(uniform) > 3483:
    from adaptive_sampling import adaptive_sample
    uniform = adaptive_sample(xyz, 3483)

elif len(uniform) < 3483:
    print(
        f"Warning: uniform baseline has only {len(uniform)} points; "
        "compare at a matched budget before drawing conclusions."
    )

start = time.perf_counter()

reduced = adaptive_sample_self_adaptive_grid(
    xyz,
    target_points=3483,
    voxel_size=0.8,
    ransac_iterations=15,
)

uniform_distances, _ = cKDTree(uniform).query(
    xyz, k=1, workers=-1
)

print(
    f"Uniform mean error: {uniform_distances.mean():.4f} m"
)
print(
    f"Uniform P95 error: "
    f"{np.percentile(uniform_distances, 95):.4f} m"
)

elapsed = time.perf_counter() - start

# Final output reconstruction error
distances, _ = cKDTree(reduced).query(xyz, k=1, workers=-1)

print(f"Final mean reconstruction error: {distances.mean():.4f} m")
print(f"Final P95 reconstruction error: {np.percentile(distances, 95):.4f} m")
print(f"Reduced points: {len(reduced)}")
print(f"Runtime: {elapsed:.3f} seconds")
print(f"All coordinates finite: {np.isfinite(reduced).all()}")
print(f"Unique output points: {len(np.unique(reduced, axis=0))}")