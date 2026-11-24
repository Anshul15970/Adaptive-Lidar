from pathlib import Path
import numpy as np
from hybrid_sampling_v4 import hybrid_sample_v4

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "object_detection" / "training" / "velodyne"
OUTPUT = ROOT / "data" / "detector_sampling"
UNIFORM_DIR = OUTPUT / "uniform" / "velodyne"
HYBRID_DIR = OUTPUT / "hybrid_v4" / "velodyne"

UNIFORM_DIR.mkdir(parents=True, exist_ok=True)
HYBRID_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
files = sorted(SOURCE.glob("*.bin"))

if not files:
    raise FileNotFoundError(f"No .bin files found in {SOURCE}")

for path in files:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        print(f"Skipping invalid file: {path.name}")
        continue

    raw = raw.reshape(-1, 4)
    xyz = raw[:, :3]
    target = len(raw) // 5

    # Uniform random point sampling without replacement.
    indices = np.sort(rng.choice(len(raw), size=target, replace=False))
    uniform_xyzi = raw[indices]

    # Hybrid V4 selects original XYZ points, so recover their XYZI rows.
    hybrid_xyz = hybrid_sample_v4(
        xyz, target_points=target, voxel_size=0.8
    )
    lookup = {}
    for row in raw:
        lookup.setdefault(tuple(row[:3]), row)

    missing = [tuple(p) for p in hybrid_xyz if tuple(p) not in lookup]
    if missing:
        raise ValueError(
            f"{path.name}: Hybrid V4 returned non-original XYZ points."
        )

    hybrid_xyzi = np.asarray(
        [lookup[tuple(p)] for p in hybrid_xyz], dtype=np.float32
    ).reshape(-1, 4)

    uniform_xyzi.tofile(UNIFORM_DIR / path.name)
    hybrid_xyzi.tofile(HYBRID_DIR / path.name)

    print(
        f"{path.stem}: original={len(raw)}, "
        f"uniform={len(uniform_xyzi)}, hybrid_v4={len(hybrid_xyzi)}"
    )

print(f"\nExport complete: {OUTPUT}")
print(f"Uniform files: {len(list(UNIFORM_DIR.glob('*.bin')))}")
print(f"Hybrid V4 files: {len(list(HYBRID_DIR.glob('*.bin')))}")
