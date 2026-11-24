
from pathlib import Path
import csv

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

from hybrid_sampling_v3 import hybrid_sample_v3


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training"
VELODYNE = DATA / "velodyne"
LABELS = DATA / "label_2"
CALIB = DATA / "calib"

OUT = ROOT / "results" / "class_retention_hybrid_v3.csv"

FRAME_LIMIT = 50
VOXEL_SIZE = 0.8
CLASSES = {"Car", "Pedestrian", "Cyclist"}


def read_calibration(path):
    values = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            arr = np.fromstring(raw, sep=" ")
            if arr.size:
                values[key.strip()] = arr

    if "Tr_velo_to_cam" not in values:
        raise ValueError(f"Missing Tr_velo_to_cam in {path}")
    if "R0_rect" not in values:
        raise ValueError(f"Missing R0_rect in {path}")

    Tr = values["Tr_velo_to_cam"].reshape(3, 4)
    R0 = values["R0_rect"].reshape(3, 3)

    return R0 @ Tr[:, :3], R0 @ Tr[:, 3]


def lidar_to_camera(xyz, rotation, translation):
    return xyz @ rotation.T + translation


def read_boxes(label_path):
    boxes = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()

            if not parts or parts[0] not in CLASSES:
                continue

            # KITTI label fields:
            # class, truncation, occlusion, alpha, bbox,
            # h, w, l, x, y, z, rotation_y
            h, w, length = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            rotation_y = float(parts[14])

            boxes.append({
                "class": parts[0],
                "h": h,
                "w": w,
                "l": length,
                "location": np.array([x, y, z]),
                "rotation_y": rotation_y,
            })

    return boxes


def points_in_box(points_camera, box):
    """Return a mask of camera-frame points inside a KITTI 3D box."""
    center = box["location"].copy()

    # KITTI location is the bottom center; camera y points downward.
    center[1] -= box["h"] / 2.0

    delta = points_camera - center
    c = np.cos(box["rotation_y"])
    s = np.sin(box["rotation_y"])

    # Transform camera-frame deltas into box-local coordinates.
    local_x = c * delta[:, 0] - s * delta[:, 2]
    local_y = delta[:, 1]
    local_z = s * delta[:, 0] + c * delta[:, 2]

    return (
        (np.abs(local_x) <= box["l"] / 2.0)
        & (np.abs(local_y) <= box["h"] / 2.0)
        & (np.abs(local_z) <= box["w"] / 2.0)
    )


def uniform_downsample(xyz, voxel_size):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        xyz.astype(np.float64)
    )
    reduced = cloud.voxel_down_sample(voxel_size)
    return np.asarray(reduced.points, dtype=np.float32)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(VELODYNE.glob("*.bin"))[:FRAME_LIMIT]
    if not frames:
        raise FileNotFoundError(f"No KITTI .bin frames found in {VELODYNE}")

    rows = []

    for frame in tqdm(frames, desc="Class retention"):
        stem = frame.stem
        label_path = LABELS / f"{stem}.txt"
        calib_path = CALIB / f"{stem}.txt"

        if not label_path.exists() or not calib_path.exists():
            print(f"Skipping {stem}: label or calibration file missing")
            continue

        raw = np.fromfile(frame, dtype=np.float32).reshape(-1, 4)
        xyz = raw[:, :3]
        xyz = xyz[np.isfinite(xyz).all(axis=1)]

        uniform = uniform_downsample(xyz, VOXEL_SIZE)
        target = len(uniform)

        hybrid = hybrid_sample_v3(
            xyz,
            target_points=target,
            voxel_size=VOXEL_SIZE,
        )

        rotation, translation = read_calibration(calib_path)
        original_cam = lidar_to_camera(xyz, rotation, translation)
        uniform_cam = lidar_to_camera(uniform, rotation, translation)
        hybrid_cam = lidar_to_camera(hybrid, rotation, translation)

        boxes = read_boxes(label_path)

        for object_id, box in enumerate(boxes):
            original_mask = points_in_box(original_cam, box)
            original_count = int(original_mask.sum())

            # Objects with no original LiDAR returns cannot be evaluated.
            if original_count == 0:
                continue

            uniform_count = int(
                points_in_box(uniform_cam, box).sum()
            )
            hybrid_count = int(
                points_in_box(hybrid_cam, box).sum()
            )

            rows.append({
                "frame": stem,
                "object_id": object_id,
                "class": box["class"],
                "original_points": original_count,
                "uniform_points": uniform_count,
                "hybrid_v3_points": hybrid_count,
                "uniform_retention_pct":
                    100.0 * uniform_count / original_count,
                "hybrid_v3_retention_pct":
                    100.0 * hybrid_count / original_count,
                "target_points": target,
            })

    if not rows:
        raise RuntimeError(
            "No evaluable objects found. Check dataset paths, labels and calibration."
        )

    fieldnames = list(rows[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nMean object-box point-count retention")
    print("-" * 65)

    for cls in sorted(CLASSES):
        class_rows = [r for r in rows if r["class"] == cls]

        if not class_rows:
            print(f"{cls:12s}: no evaluable objects")
            continue

        uniform_mean = np.mean([
            r["uniform_retention_pct"] for r in class_rows
        ])
        hybrid_mean = np.mean([
            r["hybrid_v3_retention_pct"] for r in class_rows
        ])

        print(f"{cls:12s} uniform : {uniform_mean:.2f}%")
        print(f"{cls:12s} hybrid_v3: {hybrid_mean:.2f}%")
        print(f"             objects : {len(class_rows)}")

    print(f"\nDetailed results saved to: {OUT}")


if __name__ == "__main__":
    main()
