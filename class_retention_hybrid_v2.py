
from pathlib import Path
import csv

import numpy as np
import open3d as o3d
from tqdm import tqdm

from hybrid_sampling_v2 import hybrid_sample_v2


ROOT = Path(__file__).resolve().parent
VELODYNE = ROOT / "data" / "object_detection" / "training" / "velodyne"
LABELS = ROOT / "data" / "object_detection" / "training" / "label_2"
CALIB = ROOT / "data" / "object_detection" / "training" / "calib"
OUT = ROOT / "results" / "class_retention_hybrid_v2.csv"

FRAME_LIMIT = 20
VOXEL_SIZE = 0.8
CLASSES = ("Car", "Pedestrian", "Cyclist")


def read_calibration(path):
    """Read KITTI Velodyne-to-camera calibration."""
    values = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key.strip()] = np.fromstring(raw, sep=" ")
            except ValueError:
                continue

    if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
        raise ValueError(f"Missing calibration matrices in {path}")

    transform = values["Tr_velo_to_cam"].reshape(3, 4)
    rectification = values["R0_rect"].reshape(3, 3)
    return transform, rectification


def lidar_to_camera(xyz, transform, rectification):
    """Transform Velodyne XYZ points into rectified camera coordinates."""
    xyz = np.asarray(xyz, dtype=np.float64)
    camera = xyz @ transform[:, :3].T + transform[:, 3]
    return camera @ rectification.T


def read_labels(path):
    """Read valid KITTI 3D object annotations."""
    objects = []

    if not path.exists():
        return objects

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 15 or p[0] not in CLASSES:
                continue

            h, w, length = map(float, p[8:11])
            x, y, z = map(float, p[11:14])
            yaw = float(p[14])

            if min(h, w, length) <= 0:
                continue

            objects.append({
                "class": p[0],
                "location": np.array([x, y, z], dtype=np.float64),
                "h": h,
                "w": w,
                "l": length,
                "yaw": yaw,
            })

    return objects


def points_in_box(camera_xyz, obj):
    """
    Return points inside a KITTI 3D box.

    KITTI location is the bottom center; camera Y points downward.
    """
    x, y, z = obj["location"]
    h, w, length = obj["h"], obj["w"], obj["l"]
    yaw = obj["yaw"]

    center = np.array([x, y - h / 2.0, z])
    delta = camera_xyz - center

    c = np.cos(yaw)
    s = np.sin(yaw)

    local_x = c * delta[:, 0] - s * delta[:, 2]
    local_z = s * delta[:, 0] + c * delta[:, 2]
    local_y = delta[:, 1]

    return (
        (np.abs(local_x) <= w / 2.0)
        & (np.abs(local_y) <= h / 2.0)
        & (np.abs(local_z) <= length / 2.0)
    )


def evaluate_frame(original_xyz, uniform_xyz, hybrid_xyz, objects,
                   transform, rectification):
    clouds = {
        "uniform": uniform_xyz,
        "hybrid_v2": hybrid_xyz,
    }

    # Transform each cloud into the coordinate system of KITTI labels.
    camera_clouds = {
        name: lidar_to_camera(points, transform, rectification)
        for name, points in clouds.items()
    }
    original_camera = lidar_to_camera(
        original_xyz, transform, rectification
    )

    rows = []

    for obj_id, obj in enumerate(objects):
        original_mask = points_in_box(original_camera, obj)
        original_count = int(original_mask.sum())

        # Ignore objects with no LiDAR returns: their retention is undefined.
        if original_count == 0:
            continue

        for method, camera_xyz in camera_clouds.items():
            reduced_count = int(points_in_box(camera_xyz, obj).sum())

            rows.append({
                "method": method,
                "class": obj["class"],
                "object_id": obj_id,
                "original_object_points": original_count,
                "reduced_object_points": reduced_count,
                "point_count_retention_pct": (
                    100.0 * reduced_count / original_count
                ),
            })

    return rows


def main():
    files = sorted(VELODYNE.glob("*.bin"))[:FRAME_LIMIT]

    if not files:
        raise FileNotFoundError(f"No KITTI point clouds found in {VELODYNE}")

    rows = []

    for path in tqdm(files, desc="Class retention", unit="frame"):
        label_path = LABELS / f"{path.stem}.txt"
        calib_path = CALIB / f"{path.stem}.txt"

        if not label_path.exists() or not calib_path.exists():
            print(f"Skipping {path.stem}: missing labels or calibration")
            continue

        raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        xyz = raw[:, :3]

        # Mandatory uniform voxel baseline.
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        uniform = np.asarray(
            cloud.voxel_down_sample(VOXEL_SIZE).points
        )

        # Hybrid V2 gets the same output budget as uniform.
        hybrid = hybrid_sample_v2(
            xyz,
            target_points=len(uniform),
            voxel_size=VOXEL_SIZE,
        )

        transform, rectification = read_calibration(calib_path)
        objects = read_labels(label_path)

        frame_rows = evaluate_frame(
            xyz, uniform, hybrid, objects, transform, rectification
        )

        for row in frame_rows:
            row["frame"] = path.stem
        rows.extend(frame_rows)

    if not rows:
        raise RuntimeError(
            "No annotated objects with LiDAR points were evaluated. "
            "Check the KITTI paths, labels, and calibration files."
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nMean object-box point-count retention by class:")
    for cls in CLASSES:
        for method in ("uniform", "hybrid_v2"):
            subset = [
                r for r in rows
                if r["class"] == cls and r["method"] == method
            ]

            if not subset:
                print(f"{cls:12s} {method:12s}: no eligible objects")
                continue

            retention = np.mean([
                r["point_count_retention_pct"] for r in subset
            ])
            print(
                f"{cls:12s} {method:12s}: "
                f"{retention:.2f}% "
                f"({len(subset)} object instances)"
            )

    print(f"\nDetailed per-object results saved to: {OUT}")


if __name__ == "__main__":
    main()
