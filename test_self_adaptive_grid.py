
import time
import numpy as np

from adaptive_sampling_self_adaptive_grid import (
    adaptive_sample_self_adaptive_grid
)

frame_path = "data/object_detection/training/velodyne/000000.bin"

points = np.fromfile(frame_path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

target_points = 3483  # Match the budget from your earlier single-frame test

print("Original points:", len(xyz))

start = time.perf_counter()

reduced = adaptive_sample_self_adaptive_grid(
    xyz,
    target_points=target_points
)

elapsed = time.perf_counter() - start

print("Reduced points:", len(reduced))
print(f"Reduction: {(1 - len(reduced) / len(xyz)) * 100:.2f}%")
print(f"Runtime: {elapsed:.3f} seconds")
print("All coordinates finite:", np.isfinite(reduced).all())
print("Unique output points:", len(np.unique(reduced, axis=0)))
