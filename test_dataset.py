from pathlib import Path
import numpy as np

data_dir = Path("data/raw")

files = list(data_dir.rglob("*.bin"))

print("LiDAR frames found:", len(files))

if files:
    points = np.fromfile(files[0], dtype=np.float32)

    if points.size % 4 != 0:
        raise ValueError("Unexpected KITTI point-cloud format")

    points = points.reshape(-1, 4)

    print("File:", files[0])
    print("Point count:", len(points))
    print("First 5 points:\n", points[:5])
    print("All values finite:", np.isfinite(points).all())
else:
    print("No .bin files found. Check your extraction folder.")