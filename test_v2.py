
from pathlib import Path
import time
import numpy as np
import open3d as o3d

from adaptive_sampling_v2 import adaptive_sample_v2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training" / "velodyne"

frame = DATA / "000000.bin"
points = np.fromfile(frame, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(xyz)
uniform = np.asarray(
    cloud.voxel_down_sample(0.8).points
)

start = time.perf_counter()
adaptive = adaptive_sample_v2(xyz, len(uniform))
elapsed = time.perf_counter() - start

print("Original points:", len(xyz))
print("Uniform points:", len(uniform))
print("Adaptive V2 points:", len(adaptive))
print("Adaptive V2 time:", round(elapsed, 3), "seconds")
