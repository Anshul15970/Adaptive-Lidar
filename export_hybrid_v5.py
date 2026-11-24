from pathlib import Path
import numpy as np
from hybrid_sampling_v5 import hybrid_sample_v5

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/object_detection/training/velodyne"
OUTPUT = ROOT / "data/detector_sampling/hybrid_v5/velodyne"

OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE.glob("*.bin"))

if not files:
    raise FileNotFoundError(f"No .bin files found in {SOURCE}")

print(f"Source frames: {len(files)}")
print(f"Output directory: {OUTPUT}")

for i, path in enumerate(files, start=1):
    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 4:
        print(f"Skipping invalid file: {path.name}")
        continue

    raw = raw.reshape(-1, 4)
    xyz = raw[:, :3]
    target = len(raw) // 5

    sampled_xyz = hybrid_sample_v5(
        xyz,
        target_points=target,
        voxel_size=0.8,
        random_state=0,
    )

    # Map selected original XYZ points back to their original XYZI rows.
    lookup = {}
    for row in raw:
        lookup.setdefault(tuple(row[:3]), row)

    missing = [
        tuple(point) for point in sampled_xyz
        if tuple(point) not in lookup
    ]
    if missing:
        raise ValueError(
            f"{path.name}: V5 returned non-original XYZ points."
        )

    sampled_xyzi = np.asarray(
        [lookup[tuple(point)] for point in sampled_xyz],
        dtype=np.float32,
    ).reshape(-1, 4)

    if len(sampled_xyzi) != target:
        raise ValueError(
            f"{path.name}: expected {target} points, "
            f"got {len(sampled_xyzi)}"
        )

    sampled_xyzi.tofile(OUTPUT / path.name)

    if i % 100 == 0 or i == len(files):
        print(f"[{i}/{len(files)}] Exported {path.name}")

print("\nV5 export complete.")
print(f"Source frames: {len(files)}")
print(f"Exported frames: {len(list(OUTPUT.glob('*.bin')))}")
print(f"Output: {OUTPUT}")
