
import numpy as np
import open3d as o3d
from adaptive_sampling import adaptive_sample

path = "data/raw/velodyne_points/data/0000000000.bin"

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

original = o3d.geometry.PointCloud()
original.points = o3d.utility.Vector3dVector(xyz)

uniform = original.voxel_down_sample(0.8)
adaptive = adaptive_sample(xyz, target_points=len(uniform.points))

print("Original:", len(xyz))
print("Uniform:", len(uniform.points))
print("Adaptive:", len(adaptive))


def chamfer_metrics(reference, reduced):
    tree = o3d.geometry.KDTreeFlann(reduced)
    distances = []

    for point in np.asarray(reference.points):
        _, _, d2 = tree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(d2[0]))

    return np.mean(distances), np.percentile(distances, 95)


adaptive_pcd = o3d.geometry.PointCloud()
adaptive_pcd.points = o3d.utility.Vector3dVector(adaptive)

uniform_mean, uniform_p95 = chamfer_metrics(original, uniform)
adaptive_mean, adaptive_p95 = chamfer_metrics(original, adaptive_pcd)

print("\nMean reconstruction error:")
print("Uniform:", uniform_mean)
print("Adaptive:", adaptive_mean)

print("\n95th percentile error:")
print("Uniform:", uniform_p95)
print("Adaptive:", adaptive_p95)
