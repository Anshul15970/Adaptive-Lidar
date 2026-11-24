
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from adaptive_sampling import adaptive_sample

path = "data/raw/velodyne_points/data/0000000000.bin"
points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
uniform = np.asarray(pcd.voxel_down_sample(0.8).points)
adaptive = adaptive_sample(xyz, target_points=len(uniform))

def neighborhood_recall(original, reduced, k=10):
    rng = np.random.default_rng(42)
    ids = rng.choice(len(original), min(5000, len(original)),
                     replace=False)
    sample = original[ids]

    original_tree = cKDTree(original)
    reduced_tree = cKDTree(reduced)

    _, original_neighbors = original_tree.query(sample, k=k + 1)
    _, reduced_neighbors = reduced_tree.query(sample, k=k)

    scores = []

    for i in range(len(sample)):
        # Original neighbours, excluding the query point
        neighbors = original[original_neighbors[i, 1:]]

        # A neighbour is preserved if a reduced point lies
        # within a small tolerance of its original position.
        distances, _ = reduced_tree.query(neighbors, k=1)
        tolerance = 0.3

        scores.append(np.mean(distances <= tolerance))

    return float(np.mean(scores))

print("Uniform neighborhood recall:",
      neighborhood_recall(xyz, uniform))
print("Adaptive neighborhood recall:",
      neighborhood_recall(xyz, adaptive))
