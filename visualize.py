import numpy as np
import open3d as o3d

path = "data/raw/velodyne_points/data/0000000000.bin"

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

o3d.visualization.draw_geometries([pcd])