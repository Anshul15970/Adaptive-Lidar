import numpy as np
from features import compute_features

path = "data/raw/velodyne_points/data/0000000000.bin"

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

# Use a subset first to test speed and correctness
sample = xyz[:10000]
scores = compute_features(sample)

print("Points:", len(scores))
print("Minimum complexity:", scores.min())
print("Maximum complexity:", scores.max())
print("Mean complexity:", scores.mean())
print("All finite:", np.isfinite(scores).all())