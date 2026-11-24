from pathlib import Path
import numpy as np

ROOT = Path("/mnt/e/Projects/adaptive-lidar")
ORIGINAL = ROOT / "data/object_detection/training/velodyne"
SAMPLED = ROOT / "data/detector_sampling/hybrid_v5/velodyne"
SPLIT = Path.home() / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"

VOXEL_SIZE = 0.8
RATIO = 0.20

frame_ids = [x.strip() for x in SPLIT.read_text().splitlines() if x.strip()]

spread_branch = 0
allocation_branch = 0

print(f"{'Frame':<12} {'Original':>10} {'Target':>10} {'Voxels':>10} {'Branch':<22}")
print("-" * 70)

for frame_id in frame_ids[:50]:
    original_path = ORIGINAL / f"{frame_id}.bin"
    sampled_path = SAMPLED / f"{frame_id}.bin"

    if not original_path.exists() or not sampled_path.exists():
        continue

    points = np.fromfile(original_path, dtype=np.float32).reshape(-1, 4)
    sampled = np.fromfile(sampled_path, dtype=np.float32).reshape(-1, 4)

    xyz = points[:, :3]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]

    target = len(sampled)
    voxel_keys = np.floor(xyz / VOXEL_SIZE).astype(np.int32)
    voxel_count = len(np.unique(voxel_keys, axis=0))

    if target <= voxel_count:
        branch = "spatial representatives"
        spread_branch += 1
    else:
        branch = "representatives + extras"
        allocation_branch += 1

    print(
        f"{frame_id:<12} {len(xyz):>10} {target:>10} "
        f"{voxel_count:>10} {branch:<22}"
    )

print("\nSUMMARY")
print("Spatial representative branch:", spread_branch)
print("Representatives + extras branch:", allocation_branch)
