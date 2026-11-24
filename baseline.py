import numpy as np
import open3d as o3d

path = "data/raw/velodyne_points/data/0000000000.bin"

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

voxel_size = 0.8
reduced = pcd.voxel_down_sample(voxel_size)

print("Original points:", len(xyz))
print("Reduced points:", len(reduced.points))
print("Reduction (%):", round(
    (1 - len(reduced.points) / len(xyz)) * 100, 2
))

o3d.visualization.draw_geometries([reduced])