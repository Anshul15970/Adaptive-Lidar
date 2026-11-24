
from pathlib import Path
import numpy as np

from hybrid_sampling_v4 import hybrid_sample_v4

frame = Path(
    "data/object_detection/training/velodyne/000000.bin"
)

raw = np.fromfile(frame, dtype=np.float32).reshape(-1, 4)
xyz = raw[:, :3]

target = 9416
result = hybrid_sample_v4(
    xyz,
    target_points=target,
    voxel_size=0.8,
)

assert len(result) == min(target, len(xyz))
assert result.shape[1] == 3
assert np.isfinite(result).all()

# Every output point should correspond to an input point.
# Exact row comparison is safe because this sampler returns input rows.
input_rows = {tuple(p) for p in xyz}
assert all(tuple(p) in input_rows for p in result)

print("Output shape:", result.shape)
print("All checks passed.")
