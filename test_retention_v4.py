
import numpy as np
from pathlib import Path
from adaptive_sampling import adaptive_sample_v4


def load_kitti_labels(label_path):
    objects = []

    with open(label_path, "r") as f:
        for line in f:
            fields = line.strip().split()

            if len(fields) < 15:
                continue

            cls = fields[0]

            # KITTI 3D bounding box dimensions and location
            h, w, length = map(float, fields[8:11])
            x, y, z = map(float, fields[11:14])
            rotation_y = float(fields[14])

            if cls not in {"Car", "Pedestrian", "Cyclist"}:
                continue

            objects.append({
                "class": cls,
                "h": h,
                "w": w,
                "length": length,
                "x": x,
                "y": y,
                "z": z,
                "rotation_y": rotation_y,
            })

    return objects


def points_in_kitti_box(points, obj):
    # KITTI labels use camera coordinates.
    # LiDAR-to-camera conversion is required before this test.
    raise NotImplementedError(
        "Transform LiDAR points into camera coordinates using "
        "calib.txt, then test the 3D bounding box."
    )


def test_retention(label_path, original_points, sampled_points):
    objects = load_kitti_labels(label_path)

    rows = []

    for obj in objects:
        original_mask = points_in_kitti_box(original_points, obj)
        sampled_mask = points_in_kitti_box(sampled_points, obj)

        before = int(original_mask.sum())
        after = int(sampled_mask.sum())

        retention = 100.0 * after / before if before else 0.0

        rows.append({
            "class": obj["class"],
            "original_points": before,
            "retained_points": after,
            "retention_pct": retention,
        })

    for row in rows:
        print(row)

    return rows


if __name__ == "__main__":
    # Replace these paths with your KITTI frame.
    frame = "000000"

    velodyne_path = Path(
        f"data/raw/velodyne_points/data/{frame}.bin"
    )
    label_path = Path(
        f"data/raw/label_2/{frame}.txt"
    )

    points = np.fromfile(
        velodyne_path, dtype=np.float32
    ).reshape(-1, 4)[:, :3]

    sampled = adaptive_sample_v4(
        points, target_points=9416
    )

    test_retention(label_path, points, sampled)
