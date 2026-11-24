
import numpy as np
import open3d as o3d
from adaptive_sampling import adaptive_sample

path = "data/raw/velodyne_points/data/0000000000.bin"

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

reduced = adaptive_sample(xyz, target_points=9416, k=20)

print("Original points:", len(xyz))
print("Adaptive points:", len(reduced))
print("Reduction (%):", round(
    (1 - len(reduced) / len(xyz)) * 100, 2
))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(reduced)
o3d.visualization.draw_geometries([pcd])
