
from pathlib import Path
import numpy as np

from hybrid_sampling_v3 import hybrid_sample_v3

frame = Path(
    "data/object_detection/training/velodyne/000000.bin"
)

raw = np.fromfile(frame, dtype=np.float32).reshape(-1, 4)
xyz = raw[:, :3]

result = hybrid_sample_v3(
    xyz,
    target_points=9416,
    voxel_size=0.8,
)

print("Output shape:", result.shape)
assert len(result) == min(9416, len(xyz))
assert np.isfinite(result).all()
print("Test passed.")
