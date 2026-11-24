
from pathlib import Path
import numpy as np
import open3d as o3d

from hybrid_sampling_v4 import hybrid_sample_v4 as optimized
from hybrid_sampling_v4_before_profile_opt import (
    hybrid_sample_v4 as baseline,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training" / "velodyne"

for path in sorted(DATA.glob("*.bin"))[:20]:
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    target = len(cloud.voxel_down_sample(0.8).points)

    old = baseline(xyz, target_points=target, voxel_size=0.8)
    new = optimized(xyz, target_points=target, voxel_size=0.8)

    # Every output row should match an original XYZ point.
    old_match = np.any(
        np.all(old[:, None, :] == xyz[None, :, :], axis=2),
        axis=1,
    )
    new_match = np.any(
        np.all(new[:, None, :] == xyz[None, :, :], axis=2),
        axis=1,
    )

    # Compare output sets without depending on output order.
    old_rows = np.unique(old, axis=0)
    new_rows = np.unique(new, axis=0)

    same_set = (
        len(old_rows) == len(new_rows)
        and np.array_equal(old_rows, new_rows)
    )

    print(
        f"{path.stem}: "
        f"baseline={len(old)}, optimized={len(new)}, "
        f"same_selected_points={same_set}, "
        f"original_points_only={old_match.all() and new_match.all()}"
    )
