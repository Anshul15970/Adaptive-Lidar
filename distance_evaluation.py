
import numpy as np
import open3d as o3d
from adaptive_sampling import adaptive_sample
from scipy.spatial import cKDTree

path = "data/raw/velodyne_points/data/0000000000.bin"
points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

uniform = np.asarray(pcd.voxel_down_sample(0.8).points)
adaptive = adaptive_sample(xyz, target_points=len(uniform))

bands = [(0, 10), (10, 20), (20, 40), (40, 1000)]

for name, reduced in [("Uniform", uniform), ("Adaptive", adaptive)]:
    tree = cKDTree(reduced)
    distances, _ = tree.query(xyz, k=1)
    ranges = np.linalg.norm(xyz, axis=1)

    print(f"\n{name}")
    for low, high in bands:
        mask = (ranges >= low) & (ranges < high)

        if mask.any():
            print(
                f"{low}-{high} m: "
                f"points={mask.sum()}, "
                f"mean_error={distances[mask].mean():.4f} m, "
                f"p95_error={np.percentile(distances[mask], 95):.4f} m"
            )
