
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d

from adaptive_sampling_v3 import adaptive_sample_v3

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training" / "velodyne"

path = sorted(DATA.glob("*.bin"))[0]
xyz = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(xyz)
uniform = np.asarray(cloud.voxel_down_sample(0.8).points)

start = perf_counter()
v3 = adaptive_sample_v3(xyz, len(uniform))
elapsed = perf_counter() - start

print("Frame:", path.stem)
print("Original points:", len(xyz))
print("Uniform points:", len(uniform))
print("V3 points:", len(v3))
print("V3 runtime:", round(elapsed, 3), "seconds")
